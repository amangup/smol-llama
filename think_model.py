from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn.functional as F
import torch.nn as nn


FLASH_ATTN_AVAILABLE = hasattr(torch.nn.functional, 'scaled_dot_product_attention')


@dataclass
class ThinkModelConfig:
    vocab_size: int
    d_model: int = 576
    d_head: int = 64
    d_cross_attn_head: int = 64
    d_mlp_proj: int = 1536

    think_d_model: int = 576
    think_d_head: int = 64
    think_d_mlp_proj: int = 1536

    n_kv_heads: int = 3
    n_attn_heads: int = 9
    n_cross_attn_heads: int = 9
    n_generate_layers: int = 12

    n_think_kv_heads: int = 3
    n_think_attn_heads: int = 9
    n_think_layers: int = 30

    rms_norm_eps: float = 1e-5

    rope_theta: float = 100000.0

    think_initializer_range: float = 0.02
    generate_initializer_range: float = 0.02

    think_seq_prefix_ratio: float = 1.0
    thought_embedding_init_normal: bool = True

    train_recurrence: int = 4
    #recurrence_loss_factor: float = 0.5
    
    padding_idx: Optional[int] = None


class Rotary(nn.Module):
    def __init__(self, config, is_think_network=False):
        super(Rotary, self).__init__()
        
        d_head = config.think_d_head if is_think_network else config.d_head
        
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, d_head, 2).float() / d_head))
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
    def __init__(self, config, is_think_network=False, is_cross_attn=False, is_causal=True):
        super(GroupedQueryAttention, self).__init__()
        q_dim = config.d_model
        kv_dim = config.think_d_model if is_cross_attn else config.d_model

        n_q_heads = config.n_cross_attn_heads if is_cross_attn else config.n_attn_heads
        n_kv_heads = config.n_cross_attn_heads if is_cross_attn else config.n_kv_heads

        d_model = config.think_d_model if is_think_network else config.d_model
        d_head = config.think_d_head if is_think_network else config.d_head
        
        self.q_proj = nn.Linear(q_dim, n_q_heads * d_head, bias=False)
        self.k_proj = nn.Linear(kv_dim, n_kv_heads * d_head, bias=False)
        self.v_proj = nn.Linear(kv_dim, n_kv_heads * d_head, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.config = config
        self.is_causal = is_causal
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

        q = self.q_proj(x)
        k = self.k_proj(context)
        v = self.v_proj(context)

        # Shape to (b_size, n_heads or n_kv_heads, seq_len, d_head)
        q = q.view(q_b_size, q_seq_len, -1, self.config.d_head).transpose(1, 2)
        k = k.view(kv_b_size, kv_seq_len, -1, self.config.d_head).transpose(1, 2)
        v = v.view(kv_b_size, kv_seq_len, -1, self.config.d_head).transpose(1, 2)

        q, k = self._apply_rotary_pos_emb(q, k, cos, sin, context_cos, context_sin)

        if FLASH_ATTN_AVAILABLE:
            out = F.scaled_dot_product_attention(q, k, v, scale=self.attn_scale, is_causal=self.is_causal, enable_gqa=True)
        else:
            # GQA
            # for k, v, match size of dim=-3 to be equal to n_attn_heads (up from n_kv_heads)
            k = k.repeat_interleave(self.config.n_attn_heads / self.config.n_kv_heads, -3)
            v = v.repeat_interleave(self.config.n_attn_heads / self.config.n_kv_heads, -3)

            qk_scaled = q @ k.transpose(-2, -1) * self.attn_scale

            attn_bias = torch.zeros(q_seq_len, kv_seq_len, dtype=q.dtype)
            if self.is_causal:
                temp_mask = torch.ones(q_seq_len, kv_seq_len, dtype=torch.bool).tril(diagonal=0)
                attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))

            attn = qk_scaled + attn_bias
            attn = F.softmax(attn, dim=-1)

            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(q_b_size, q_seq_len, -1)
        return self.o_proj(out)


class GatedMlp(nn.Module):
    def __init__(self, config, is_think_network=False):
        super(GatedMlp, self).__init__()

        d_model = config.think_d_model if is_think_network else config.d_model
        d_mlp_proj = config.think_d_mlp_proj if is_think_network else config.d_mlp_proj
        
        self.up_proj = nn.Linear(d_model, d_mlp_proj, bias=False)
        self.gate_proj = nn.Linear(d_model, d_mlp_proj, bias=False)
        self.down_proj = nn.Linear(d_mlp_proj, d_model, bias=False)
        self.silu = nn.SiLU()


    def forward(self, x):
        up = self.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(up)


class ThinkLayer(nn.Module):
    def __init__(self, config, is_causal=True):
        super(ThinkLayer, self).__init__()

        is_causal=True
        if config.think_seq_prefix_ratio < 1.0:
            is_causal=False

        self.self_attn = GroupedQueryAttention(config, is_think_network=True, is_causal=is_causal)
        self.mlp = GatedMlp(config, is_think_network=True)

        self.input_layernorm = nn.modules.normalization.RMSNorm(config.think_d_model, config.rms_norm_eps)
        self.post_attention_layernorm = nn.modules.normalization.RMSNorm(config.think_d_model, config.rms_norm_eps)

    def forward(self, x, cos, sin):
        x_norm = self.input_layernorm(x)
        x = x + self.self_attn(x_norm, x_norm, cos, sin, cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class CrossAttnDecoderLayer(nn.Module):
    def __init__(self, config):
        super(CrossAttnDecoderLayer, self).__init__()

        self.self_attn = GroupedQueryAttention(config)
        self.cross_attn = GroupedQueryAttention(config, is_cross_attn=True)
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
            embedding_dim=config.think_d_model)
        self.layers = nn.ModuleList([ThinkLayer(config) for _ in range(config.n_think_layers)])
        self.norm = nn.modules.normalization.RMSNorm(config.think_d_model, config.rms_norm_eps)
        self.rotary_emb = Rotary(config, is_think_network=True)

        for module in self.modules():
            std = config.think_initializer_range
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight.data, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight.data, mean=0.0, std=std)
                if config.padding_idx is not None:
                    module.weight.data[config.padding_idx].zero_()

    def forward(self, inp, think_r):
        input_embedding = self.embed_tokens(inp)

        if self.config.thought_embedding_init_normal:
            thought_embedding = torch.randn_like(input_embedding)
        else:
            thought_embedding = torch.zeros_like(input_embedding)

        thought_embeddings = []
        for _ in range(think_r):
            x = input_embedding + thought_embedding
            cos, sin = self.rotary_emb(x, seq_dim=1)

            for layer in self.layers:
                x = layer(x, cos, sin)
            thought_embedding = self.norm(x)
            thought_embeddings.append(thought_embedding)

        # shape: (batch size, seq_len, recurrence i, model dimension)
        all_thought_embeddings = torch.stack(thought_embeddings, dim=-2)
        if len(thought_embeddings) > 1:
            cosine_sim = F.cosine_similarity(
                all_thought_embeddings[:, :, :-1, :],  # All except last
                all_thought_embeddings[:, :, 1:, :],   # All except first
                dim=-1
            ).mean()
        else:
            cosine_sim = 0

        loss = cosine_sim

        return all_thought_embeddings, loss


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

        for module in self.modules():
            std = config.generate_initializer_range
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight.data, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight.data, mean=0.0, std=std)
                if config.padding_idx is not None:
                    module.weight.data[config.padding_idx].zero_()

    def forward(self, x, thought_embedding, y=None):
        x = self.embed_tokens(x)
        cos, sin = self.rotary_emb(x, seq_dim=1)
        context_cos, context_sin = self.rotary_emb(thought_embedding, seq_dim=1)
        for layer in self.layers:
            x = layer(x, thought_embedding, cos, sin, context_cos, context_sin)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if y is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-1)

        return logits, loss

class ThinkTransformer(nn.Module):
    def __init__(self, config):
        super(ThinkTransformer, self).__init__()
        self.config = config

        self.think_network = ThinkNetwork(config)
        self.generate_network = GenerateNetwork(config)


    def _reshape_thought_embedding(self, thought_embedding, train=True):
        batch_size, seq_len, recurrence, dim = thought_embedding.shape
        train_r = self.config.train_recurrence

        if train:
            start_idx = 0
        else:
            start_idx = (seq_len % train_r) - 1
            start_idx = start_idx + train_r if start_idx < 0 else start_idx

        thought_embedding = thought_embedding[:, start_idx:seq_len:train_r, ::(recurrence // train_r), :]
        return thought_embedding.reshape(batch_size, -1, dim)[:, -seq_len:, :]

    # For training only
    # For inference, we don't run think_network at all time steps of generation.
    def forward(self, x, y=None):
        if self.config.think_seq_prefix_ratio < 1.0:
            think_seq_len = math.floor(self.config.think_seq_prefix_ratio * x.shape[1])
            think_x = x[:, :think_seq_len]
            generate_x, generate_y = x[:, think_seq_len:], y[:, think_seq_len:]
        else:
            think_x, generate_x, generate_y = x, x, y

        thought_embedding, recurrence_cosine_sim = self.think_network(think_x, think_r=self.config.train_recurrence)

        thought_embedding = self._reshape_thought_embedding(thought_embedding)

        logits, ce_loss = self.generate_network(generate_x, thought_embedding, generate_y)
        loss = ce_loss
        #loss = ce_loss + self.config.recurrence_loss_factor * recurrence_loss

        additional_metrics = {"recurrence_cosine_sim": recurrence_cosine_sim}

        return logits, loss, additional_metrics


    @torch.no_grad()
    def generate(self, idx, temperature=1.0, top_k=None, max_new_tokens=128, think_r=8):
        assert think_r % self.config.train_recurrence == 0

        for i in range(max_new_tokens):
            if i % think_r == 0:
                thought_embedding, _ = self.think_network(idx, think_r=think_r)
                thought_embedding = self._reshape_thought_embedding(thought_embedding, train=False)
            else:
                thought_embedding = torch.cat((thought_embedding, thought_embedding[:, -1:, :]), dim=1)

            logits, _, _ = self.generate_network(idx, thought_embedding)

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
