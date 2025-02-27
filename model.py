from dataclasses import dataclass
from typing import Optional

import re
import torch
import torch.nn.functional as F
import torch.nn as nn

@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 576
    d_head: int = 64
    d_mlp_proj: int = 1536

    n_kv_heads: int = 3
    n_attn_heads: int = 9
    n_layers: int = 30

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

        self.use_flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    @staticmethod
    def _rotate_half(x):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)


    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin


    def forward(self, x, cos, sin):
        b_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Shape to (b_size, n_heads or n_kv_heads, seq_len, d_head)
        q = q.view(b_size, seq_len, -1, self.config.d_head).transpose(1, 2)
        k = k.view(b_size, seq_len, -1, self.config.d_head).transpose(1, 2)
        v = v.view(b_size, seq_len, -1, self.config.d_head).transpose(1, 2)

        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        if self.use_flash:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            # GQA
            # for k, v, match size of dim=-3 to be equal to n_attn_heads (up from n_kv_heads)
            k = k.repeat_interleave(self.config.n_attn_heads / self.config.n_kv_heads, -3)
            v = v.repeat_interleave(self.config.n_attn_heads / self.config.n_kv_heads, -3)

            qk_scaled = q @ k.transpose(-2, -1) * self.attn_scale

            # causal mask
            attn_bias = torch.zeros(seq_len, seq_len, dtype=q.dtype)
            temp_mask = torch.ones(seq_len, seq_len, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn = qk_scaled + attn_bias

            attn = F.softmax(attn, dim=-1)
            out = attn @ v

        out = out.transpose(1, 2).contiguous().view(b_size, seq_len, -1)
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
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, config):
        super(LlamaModel, self).__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])
        self.norm = nn.modules.normalization.RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.rotary_emb = Rotary(config)

        for module in self.modules():
            std = config.initializer_range
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight.data, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight.data, mean=0.0, std=std)
                if config.padding_idx is not None:
                    module.weight.data[config.padding_idx].zero_()


    def forward(self, x, y=None):
        x = self.embed_tokens(x)
        cos, sin = self.rotary_emb(x, seq_dim=1)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if y is not None:
            # logits.shape is (b_size, seq_len, vocab_size)
            # y.shape is (b_size, seq_len)
            # F.cross_entropy wants logits to be (b_size, vocab_size), y to be (b_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

        return logits, loss, {}


    @torch.no_grad()
    def generate(self, idx, temperature=1.0, top_k=None, max_new_tokens=128):
        for _ in range(max_new_tokens):
            logits, _, _ = self(idx)
            logits = logits[:, -1, :] / temperature
    
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def using_flash_attention(self):
        return self.layers[0].self_attn.use_flash


def main():
    config = ModelConfig(
        vocab_size=49152,
        d_model=576,
        d_head=64,
        d_mlp_proj=1536,
        n_layers=30,
        n_kv_heads=3,
        n_attn_heads=9,
        rms_norm_eps=1e-5,
        initializer_range=0.041666666666666664,
        rope_theta=100000.0
    )

    model = LlamaModel(config)
    x = torch.randint(0, config.vocab_size, (4, 1024))
    out, _ = model(x)
    print(out.shape)


if __name__ == "__main__":
    main()
