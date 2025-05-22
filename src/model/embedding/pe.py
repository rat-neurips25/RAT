import math
import torch
import torch.nn as nn
from ...utils.registry import pe_registry


@pe_registry.register("empty")
class Empty(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, seq_start, seq_end, device, dtype):
        return None, {}

    def step(self, seq_start, seq_end, device, dtype):
        return None, {}


@pe_registry.register("interrope")
class InterRoPE(nn.Module):
    # rope across chunks
    def __init__(self, dim, max_num_chunk, chunk_size, base=10000, device=None, **kwargs):
        super().__init__()
        self.dim = dim
        self.max_num_chunk = max_num_chunk
        self.chunk_size = chunk_size
        self.base = base
        inv_freq_rope = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq_rope", inv_freq_rope, persistent=False)
        self._set_cos_sin_cache_rope(num_chunk=max_num_chunk, device=inv_freq_rope.device, dtype=inv_freq_rope.dtype)

    def _set_cos_sin_cache_rope(self, num_chunk, device, dtype):
        t = torch.arange(num_chunk, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq_rope)
        rope_emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", rope_emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", rope_emb.sin().to(dtype), persistent=False)

    def forward(self, seq_start, seq_end, device, dtype):
        end_id = (seq_end - 1) // self.chunk_size
        if end_id >= self.max_num_chunk:
            self.max_num_chunk = end_id + 1
            self._set_cos_sin_cache_rope(self.max_num_chunk, device, dtype)
        cos = self.cos_cached.unsqueeze(-2).repeat(1, self.chunk_size, 1).flatten(0, 1)
        sin = self.sin_cached.unsqueeze(-2).repeat(1, self.chunk_size, 1).flatten(0, 1)
        return None, {f"interrope_{self.chunk_size}": (cos[seq_start: seq_end], sin[seq_start: seq_end])}

    def step(self, seq_start, seq_end, device, dtype):
        end_id = (seq_end - 1) // self.chunk_size
        if end_id >= self.max_num_chunk:
            self.max_num_chunk = end_id + 1
            self._set_cos_sin_cache_rope(self.max_num_chunk, device, dtype)
        return None, {f"interrope_{self.chunk_size}": (self.cos_cached[end_id: end_id + 1].to(device=device, dtype=dtype),
                                         self.sin_cached[end_id: end_id + 1].to(device=device, dtype=dtype))}


@pe_registry.register("rope")
class RoPE(nn.Module):

    def __init__(self, dim, max_seq_len=2048, base=10000, device="cuda", **kwargs):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, seq_start, seq_end, device, dtype): # [seq_start, seq_end)
        if seq_end > self.max_seq_len:
            self.max_seq_len = seq_end
            self._set_cos_sin_cache(self.max_seq_len, device, dtype)
        return None, \
            {"rope": (self.cos_cached[seq_start: seq_end].to(device=device, dtype=dtype),
                                self.sin_cached[seq_start: seq_end].to(device=device, dtype=dtype))}

    def step(self, seq_start, seq_end, device, dtype):
        return self.forward(seq_start, seq_end, device, dtype)