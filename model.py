"""Pion — Dense Transformer LM with GQA, QK-Norm, Logit Soft-Capping, Tied Embeddings"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class PionConfig:
    vocab_size: int = 50258
    d_model: int = 640
    n_layers: int = 16
    n_heads: int = 10
    n_kv_heads: int = 2
    d_ff: int = 1792
    max_seq_len: int = 2048
    dropout: float = 0.0
    rope_theta: float = 10000.0
    logit_cap: float = 30.0
    use_soft_cap: bool = True
    tie_embeddings: bool = True

    @classmethod
    def nano(cls):
        return cls(d_model=384, n_layers=6, n_heads=6, n_kv_heads=2, d_ff=1024)

    @classmethod
    def micro(cls):
        return cls(d_model=512, n_layers=12, n_heads=8, n_kv_heads=2, d_ff=1408)

    @classmethod
    def small(cls):
        return cls()

    @classmethod
    def base(cls):
        return cls(d_model=768, n_layers=20, n_heads=12, n_kv_heads=4, d_ff=2048)

    @classmethod
    def large(cls):
        return cls(d_model=1024, n_layers=24, n_heads=16, n_kv_heads=4, d_ff=2816)

    @classmethod
    def xl(cls):
        return cls(d_model=1536, n_layers=28, n_heads=16, n_kv_heads=4, d_ff=4096)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.register_buffer("inv_freq", None, persistent=False)
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        self._cached_seq_len = 0

    def _build_cache(self, seq_len, device, dtype):
        if self.inv_freq is None or self.inv_freq.device != device:
            inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        self._cached_seq_len = seq_len

    def forward(self, seq_len, device, dtype):
        if self.cos_cached is None or seq_len > self._cached_seq_len or self.cos_cached.device != device:
            self._build_cache(seq_len, device, dtype)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


class KVCache:
    def __init__(self):
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def update(self, k, v):
        if self.k_cache is None:
            self.k_cache = k
            self.v_cache = v
        else:
            self.k_cache = torch.cat([self.k_cache, k], dim=2)
            self.v_cache = torch.cat([self.v_cache, v], dim=2)
        return self.k_cache, self.v_cache

    def reset(self):
        self.k_cache = None
        self.v_cache = None

    @property
    def seq_len(self):
        return 0 if self.k_cache is None else self.k_cache.size(2)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: PionConfig, layer_idx: int):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.logit_cap = config.logit_cap
        self.use_soft_cap = config.use_soft_cap
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.rotary = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x, mask=None, kv_cache=None):
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cache_offset = kv_cache.seq_len if kv_cache is not None else 0
        total_len = cache_offset + seq_len
        cos, sin = self.rotary(total_len, x.device, x.dtype)
        q = apply_rotary_emb(q, cos[cache_offset:total_len], sin[cache_offset:total_len])
        k = apply_rotary_emb(k, cos[cache_offset:total_len], sin[cache_offset:total_len])

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        k_expanded = k.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else k
        v_expanded = v.repeat_interleave(self.n_rep, dim=1) if self.n_rep > 1 else v

        if self.logit_cap > 0 and self.use_soft_cap and self.training:
            scores = torch.matmul(q, k_expanded.transpose(-2, -1)) * self.scale
            scores = self.logit_cap * torch.tanh(scores / self.logit_cap)
            kv_len = k_expanded.size(2)
            causal = torch.triu(torch.ones(seq_len, kv_len, device=x.device, dtype=torch.bool), diagonal=kv_len - seq_len + 1)
            scores.masked_fill_(causal, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v_expanded)
        else:
            out = F.scaled_dot_product_attention(
                q, k_expanded, v_expanded,
                attn_mask=mask,
                is_causal=mask is None and kv_cache is None,
            )

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, config: PionConfig, layer_idx: int):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(config, layer_idx)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x, mask=None, kv_cache=None):
        x = x + self.attn(self.norm1(x), mask, kv_cache)
        x = x + self.ffn(self.norm2(x))
        return x


class Pion(nn.Module):
    def __init__(self, config: PionConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([TransformerBlock(config, i) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_kv_caches(self):
        return [KVCache() for _ in range(len(self.layers))]

    def forward(self, input_ids, labels=None, mask=None, kv_caches=None):
        x = self.embed(input_ids)

        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            x = layer(x, mask, kv_cache)

        x = self.norm(x)

        loss = None
        if labels is not None:
            # Chunked CE: apply lm_head per chunk to avoid materializing full logits tensor
            shift_hidden = x[:, :-1, :].contiguous().view(-1, x.size(-1))
            shift_labels = labels[:, 1:].contiguous().view(-1)
            chunk_size = 2048
            total_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            for i in range(0, shift_hidden.shape[0], chunk_size):
                chunk_logits = self.lm_head(shift_hidden[i:i + chunk_size])
                chunk_loss = F.cross_entropy(
                    chunk_logits,
                    shift_labels[i:i + chunk_size],
                    ignore_index=-100,
                    reduction="sum",
                )
                total_loss = total_loss + chunk_loss
            valid_tokens = (shift_labels != -100).sum()
            loss = total_loss / valid_tokens.clamp(min=1)
            logits = None
        else:
            logits = self.lm_head(x)

        return {"logits": logits, "loss": loss}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def get_model(size="small", **overrides):
    configs = {
        "nano": PionConfig.nano,
        "micro": PionConfig.micro,
        "small": PionConfig.small,
        "base": PionConfig.base,
        "large": PionConfig.large,
        "xl": PionConfig.xl,
    }
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")
    config = configs[size]()
    for k, v in overrides.items():
        setattr(config, k, v)
    return Pion(config)


def fmt(n):
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.1f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return str(int(n))


if __name__ == "__main__":
    for name in ["nano", "micro", "small", "base", "large", "xl"]:
        config = getattr(PionConfig, name)()
        model = Pion(config)
        params = model.count_parameters()
        print(f"Pion-{name:5s}: {fmt(params):>6s} params | d={config.d_model} L={config.n_layers} H={config.n_heads}/{config.n_kv_heads}")

    print("\nForward pass test (nano):")
    config = PionConfig.nano()
    model = Pion(config)
    x = torch.randint(0, config.vocab_size, (2, 128))
    out = model(x, labels=x)
    print(f"  Loss: {out['loss'].item():.4f}")
    print(f"  Logits during training: {out['logits']}")
    out_inf = model(x)
    print(f"  Logits during inference: {out_inf['logits'].shape}")

    print("\nKV-cache test:")
    model.eval()
    kv_caches = model.create_kv_caches()
    with torch.no_grad():
        prompt = torch.randint(0, config.vocab_size, (1, 10))
        out1 = model(prompt, kv_caches=kv_caches)
        print(f"  Prompt: {prompt.shape}, Cache: {kv_caches[0].seq_len}")
        next_tok = torch.randint(0, config.vocab_size, (1, 1))
        out2 = model(next_tok, kv_caches=kv_caches)
        print(f"  Next: {next_tok.shape}, Cache: {kv_caches[0].seq_len}")
    print("Done!")
