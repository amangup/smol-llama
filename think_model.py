from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
import torch.nn as nn


FLASH_ATTN_AVAILABLE = hasattr(torch.nn.functional, 'scaled_dot_product_attention')


@dataclass
class ThinkModelConfig:
    vocab_size: int
    d_model: int = 576
    d_head: int = 64
    d_mlp_proj: int = 1536

    n_kv_heads: int = 3
    n_attn_heads: int = 9
    n_think_layers: int = 30
    n_generate_layers: int = 12

    rms_norm_eps: float = 1e-5

    rope_theta: float = 100000.0

    initializer_range: float = 0.02
    padding_idx: Optional[int] = None

    tie_word_embeddings: bool = False


class Rotary(nn.Module):
    def __init__(self, config):
        super(Rotary, self).__init__()
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.d_head, 2).float() / config.d_head))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.size(seq_dim)
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()

        return self.cos_cached, self.sin_cached


class GroupedQueryAttention(nn.Module):
    def __init__(self, config):
        super(GroupedQueryAttention, self).__init__()
        self.q_proj = nn.Linear(config.d_model, config.n_attn_heads * config.d_head, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.config = config

        self.attn_scale = config.d_head ** -0.5


    @staticmethod
    def _rotate_half(x):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rotary_pos_emb(self, q, k, q_cos, q_sin, k_cos, k_sin):
        return q * q_cos + self._rotate_half(q) * q_sin, k * k_cos + self._rotate_half(k) * k_sin

    def forward(self, x, context, cos, sin, context_cos, context_sin):
        q_b_size, q_seq_len, q_dim = x.shape
        kv_b_size, kv_seq_len, kv_dim = context.shape

        torch._assert(q_b_size == kv_b_size, f"Batch sizes must match: {q_b_size} != {kv_b_size}")
        torch._assert(q_dim == kv_dim, f"Hidden state dimensions must match: {q_dim} != {kv_dim}")

        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        # Shape to (b_size, n_heads or n_kv_heads, seq_len, d_head)
        q = q.view(q_b_size, q_seq_len, -1, self.config.d_head).transpose(1, 2)
        k = k.view(kv_b_size, kv_seq_len, -1, self.config.d_head).transpose(1, 2)
        v = v.view(kv_b_size, kv_seq_len, -1, self.config.d_head).transpose(1, 2)

        q, k = self._apply_rotary_pos_emb(q, k, cos, sin, context_cos, context_sin)

        if FLASH_ATTN_AVAILABLE:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.attn_scale, is_causal=True, enable_gqa=True)
        else:
            # GQA
            # for k, v, match size of dim=-3 to be equal to n_attn_heads (up from n_kv_heads)
            k = k.repeat_interleave(self.config.n_attn_heads / self.config.n_kv_heads, -3)
            v = v.repeat_interleave(self.config.n_attn_heads / self.config.n_kv_heads, -3)

            qk_scaled = q @ k.transpose(-2, -1) * self.attn_scale

            attn_bias = torch.zeros(q_seq_len, kv_seq_len, dtype=q.dtype)
            temp_mask = torch.ones(q_seq_len, kv_seq_len, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

            attn = qk_scaled + attn_bias
            attn = F.softmax(attn, dim=-1)

            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(q_b_size, q_seq_len, -1)
        return self.o_proj(out)


class GatedMlp(nn.Module):
    def __init__(self, config):
        super(GatedMlp, self).__init__()

        self.up_proj = nn.Linear(config.d_model, config.d_mlp_proj, bias=False)
        self.gate_proj = nn.Linear(config.d_model, config.d_mlp_proj, bias=False)
        self.down_proj = nn.Linear(config.d_mlp_proj, config.d_model, bias=False)
        self.silu = nn.SiLU()


    def forward(self, x):
        up = self.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(up)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super(DecoderLayer, self).__init__()

        self.self_attn = GroupedQueryAttention(config)
        self.mlp = GatedMlp(config)
        self.input_layernorm = nn.modules.normalization.RMSNorm(config.d_model, config.rms_norm_eps)
        self.post_attention_layernorm = nn.modules.normalization.RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(self, x, cos, sin):
        x_norm = self.input_layernorm(x)
        x = x + self.self_attn(x_norm, x_norm, cos, sin, cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class CrossAttnDecoderLayer(nn.Module):
    def __init__(self, config):
        super(CrossAttnDecoderLayer, self).__init__()

        self.self_attn = GroupedQueryAttention(config)
        self.cross_attn = GroupedQueryAttention(config)
        self.mlp = GatedMlp(config)
        self.self_attn_input_layernorm = nn.modules.normalization.RMSNorm(config.d_model, config.rms_norm_eps)
        self.cross_attn_input_layernorm = nn.modules.normalization.RMSNorm(config.d_model, config.rms_norm_eps)
        self.post_attention_layernorm = nn.modules.normalization.RMSNorm(config.d_model, config.rms_norm_eps)

    def forward(self, x, thought_embedding, cos, sin, context_cos, context_sin):
        x_norm = self.self_attn_input_layernorm(x)
        x = x + self.self_attn(x_norm, x_norm, cos, sin, cos, sin)
        x_norm = self.cross_attn_input_layernorm(x)
        x = x + self.cross_attn(x_norm, thought_embedding, cos, sin, context_cos, context_sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class ThinkNetwork(nn.Module):
    def __init__(self, config):
        super(ThinkNetwork, self).__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_think_layers)])
        self.norm = nn.modules.normalization.RMSNorm(config.d_model, config.rms_norm_eps)
        self.rotary_emb = Rotary(config)

    def forward(self, inp, think_r=2):
        input_embedding = self.embed_tokens(inp)
        thought_embedding = torch.zeros_like(input_embedding)

        for _ in range(think_r):
            x = input_embedding + thought_embedding
            cos, sin = self.rotary_emb(x, seq_dim=1)

            for layer in self.layers:
                x = layer(x, cos, sin)
            thought_embedding = self.norm(x)

        return thought_embedding


class GenerateNetwork(nn.Module):
    def __init__(self, config):
        super(GenerateNetwork, self).__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model)
        self.layers = nn.ModuleList([CrossAttnDecoderLayer(config) for _ in range(config.n_generate_layers)])
        self.norm = nn.modules.normalization.RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.rotary_emb = Rotary(config)

    def forward(self, x, thought_embedding):
        x = self.embed_tokens(x)
        cos, sin = self.rotary_emb(x, seq_dim=1)
        context_cos, context_sin = self.rotary_emb(thought_embedding, seq_dim=1)
        for layer in self.layers:
            x = layer(x, thought_embedding, cos, sin, context_cos, context_sin)
        x = self.norm(x)
        logits = self.lm_head(x)

        return logits


class ThinkTransformer(nn.Module):
    def __init__(self, config):
        super(ThinkTransformer, self).__init__()
        self.config = config

        self.think_network = ThinkNetwork(config)
        self.generate_network = GenerateNetwork(config)

        for module in self.modules():
            std = config.initializer_range
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight.data, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight.data, mean=0.0, std=std)
                if config.padding_idx is not None:
                    module.weight.data[config.padding_idx].zero_()

    # For training only
    # For inference, we don't run think_network at all time steps of generation.
    def forward(self, x, y=None, think_r=2, k: int = 8):
        thought_embedding = self.think_network(x, think_r=think_r)

        # select every kth thought embedding, and repeat it
        seq_len = thought_embedding.shape[1]
        anchor_embeddings = thought_embedding[:, ((seq_len % k) - 1):seq_len:k, :]
        thought_embedding = anchor_embeddings.repeat_interleave(k, dim=1)[:, :seq_len, :]

        logits = self.generate_network(x, thought_embedding)

        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, temperature=1.0, top_k=None, max_new_tokens=128, think_r=8, k: int = 8):
        for i in range(max_new_tokens):
            if i % k == 0:
                thought_embedding = self.think_network(idx, think_r=think_r)
            else:
                thought_embedding = torch.cat((thought_embedding, thought_embedding[:, -1:, :]), dim=1)

            logits = self.generate_network(idx, thought_embedding)

            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def using_flash_attention(self):
        return FLASH_ATTN_AVAILABLE

def main():
    pass
    # config = ModelConfig(
    #     vocab_size=49152,
    #     d_model=576,
    #     d_head=64,
    #     d_mlp_proj=1536,
    #     n_layers=30,
    #     n_kv_heads=3,
    #     n_attn_heads=9,
    #     rms_norm_eps=1e-5,
    #     initializer_range=0.041666666666666664,
    #     rope_theta=100000.0
    # )
    #
    # model = ThinkModel(config)
    # x = torch.randint(0, config.vocab_size, (4, 1024))
    # out, _ = model(x)
    # print(out.shape)


if __name__ == "__main__":
    main()
