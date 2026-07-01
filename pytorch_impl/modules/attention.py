import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class AttentionConfig:
    """Shared config for KV_cache, MHSA, CrossAttention, and GQA; each pulls only the fields it needs."""
    d_model: int
    num_heads: int = None       # used by MHSA / CrossAttention
    num_q_heads: int = None     # used by GQA (falls back to num_heads if unset)
    num_kv_heads: int = None    # used by GQA
    max_seq_len: int = None     # used by KV_cache
    num_layers: int = None      # used by KV_cache
    batch_size: int = None      # used by KV_cache

    def __post_init__(self):
        if self.num_q_heads is None:
            self.num_q_heads = self.num_heads

class KV_cache:
    """
    Pre-allocated per-layer key/value cache for autoregressive decoding.
    Holds fixed-size buffers of shape (batch_size, num_heads, max_seq_len, head_dim)
    for each layer, and a single write cursor (`pos`) shared across all layers,
    since every layer advances by the same number of new tokens per forward pass.
    """
    def __init__(self, config: AttentionConfig):
        kv_heads = config.num_kv_heads or config.num_heads
        head_dim = config.d_model // config.num_q_heads
        self.k_cache = [torch.zeros(config.batch_size, kv_heads, config.max_seq_len, head_dim) for _ in range(config.num_layers)]
        self.v_cache = [torch.zeros(config.batch_size, kv_heads, config.max_seq_len, head_dim) for _ in range(config.num_layers)]
        self.pos = 0

    def update(self, k_new, v_new, layer_num):
        """
        Writes k_new/v_new into the cache at the current cursor for the given layer,
        and returns the full k/v history up to and including this write
        (not just the newly written chunk) so attention can be computed over all past tokens.
        """
        new_seq_len = k_new.shape[2]
        self.k_cache[layer_num][:, :, self.pos: self.pos+new_seq_len, :] = k_new
        self.v_cache[layer_num][:, :, self.pos: self.pos+new_seq_len, :] = v_new
        return self.k_cache[layer_num][:, :, : self.pos+new_seq_len, :], self.v_cache[layer_num][:, :, : self.pos+new_seq_len, :]

    def step(self, n_new):
        """Advances the shared write cursor; call once per forward pass, after all layers have updated."""
        self.pos += n_new

    def reset(self):
        """Resets the cursor to start a new sequence (buffer contents are overwritten on next update, not cleared)."""
        self.pos = 0

class MHSA(nn.Module):
    """Multi-head self-attention with causal masking, optional padding mask, and optional KV cache for decoding."""
    def __init__(self, config: AttentionConfig):
        super().__init__()
        d_model, num_heads = config.d_model, config.num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, layer_num, padding_mask=None, kv_cache=None):
        """
        x: (B, L, D) input for this step — L == 1 during autoregressive decode, L > 1 during prefill.
        layer_num: index into kv_cache identifying which layer's buffer to read/write.
        padding_mask: (B, L_total) bool mask, True at key positions to ignore (e.g. padding tokens).
        kv_cache: optional KV_cache; when given, k/v are appended to cached history before attending.
        """
        B, L, D = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v, layer_num)

        L_total = k.shape[2]
        attn = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)

        combined_mask = None
        # L=1 during the decode 
        # L>1 prefill
        if L>1:
            l_new = L_total - L
            mask = ~torch.tril(torch.ones(L, L_total, device=x.device, dtype=torch.bool), diagonal=l_new)
            combined_mask = mask
        if padding_mask is not None:
            # padding mask defines which key position to ignore
            mask =  padding_mask.view(B, 1, 1, L_total)
            if combined_mask is not None:
                combined_mask = combined_mask | mask
        if combined_mask is not None:
            attn = attn.masked_fill(combined_mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        val = attn @ v
        val = torch.reshape(val.transpose(1, 2), (B, L, D))
        out = self.out(val)
        return out

        

class CrossAttention(nn.Module):
    """Multi-head cross-attention: queries come from `x`, keys/values from `context`. No causal mask."""
    def __init__(self, config: AttentionConfig):
        super().__init__()
        d_model, num_heads = config.d_model, config.num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.head_dim = d_model // num_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2*d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, context, padding_mask=None):
        """
        x: (B, L_q, D) query input. context: (B, L_k, D) key/value source (e.g. encoder output).
        padding_mask: (B, L_k) bool mask, True at context positions to ignore.
        """
        B, L_q, d = x.shape
        q = self.q(x).view(B, L_q, self.num_heads, self.head_dim).transpose(1,2)
        kv = self.kv(context)
        k, v = torch.chunk(kv, 2, dim=-1)
        L_k = context.shape[1]
        k = k.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L_k, self.num_heads, self.head_dim).transpose(1, 2)

        attn = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if padding_mask is not None:
            mask = padding_mask.view(B, 1, 1, L_k)
            attn = attn.masked_fill(mask, float('-inf'))
        attn_scores = torch.softmax(attn, dim=-1)
        val = attn_scores @ v
        val = torch.reshape(val.transpose(1, 2), (B, L_q, d))
        return self.out(val)


class GQA(nn.Module):
    """Grouped-query self-attention: fewer K/V heads than Q heads, each K/V head shared across a group of Q heads."""
    def __init__(self, config: AttentionConfig):
        super().__init__()
        d_model, num_q_heads, num_kv_heads = config.d_model, config.num_q_heads, config.num_kv_heads
        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        assert d_model % num_q_heads == 0
        self.q_head_dim = d_model // num_q_heads
        assert num_q_heads % num_kv_heads == 0
        self.kv_head_dim = self.q_head_dim
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.kv = nn.Linear(d_model, 2*num_kv_heads*self.q_head_dim, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, layer_num, padding_mask=None, kv_cache=None):
        """
        x: (B, L, D) input for this step — L == 1 during autoregressive decode, L > 1 during prefill.
        layer_num: index into kv_cache identifying which layer's buffer to read/write.
        padding_mask: (B, L_total) bool mask, True at key positions to ignore.
        kv_cache: optional KV_cache (sized for num_kv_heads); k/v are cached before being expanded to num_q_heads.
        """
        B, L, D = x.shape
        q = self.q(x)
        kv = self.kv(x)
        k, v = torch.chunk(kv, 2, dim=-1)
        q = q.view(B, L, self.num_q_heads, self.q_head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_kv_heads, self.kv_head_dim).transpose(1, 2)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, layer_num)
        k = k.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
        v = v.repeat_interleave(self.num_q_heads // self.num_kv_heads, dim=1)
        

        L_total = k.shape[2]
        attn = q @ k.transpose(-1, -2) / math.sqrt(self.q_head_dim)

        combined_mask = None
        # L=1 during the decode 
        # L>1 prefill
        if L>1:
            l_new = L_total - L
            mask = ~torch.tril(torch.ones(L, L_total, device=x.device, dtype=torch.bool), diagonal=l_new)
            combined_mask = mask
        if padding_mask is not None:
            # padding mask defines which key position to ignore
            mask =  padding_mask.view(B, 1, 1, L_total)
            if combined_mask is not None:
                combined_mask = combined_mask | mask
        if combined_mask is not None:
            attn = attn.masked_fill(combined_mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        val = attn @ v
        val = torch.reshape(val.transpose(1, 2), (B, L, D))
        out = self.out(val)
        return out
