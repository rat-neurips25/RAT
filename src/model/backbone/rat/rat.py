import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from torch.nn.attention import flex_attention as fla
from ....utils.registry import layer_registry, norm_registry
from ...base import Base
from ..util import apply_rotary_pos_emb
from ...op import pscan, merge_last_token_naive, ascan
from ..cache import RATCache
# v, out_proj, ogate, fgate (k, q shared)  Full for RNN ngroups=2 1024 * 64 * 4


@layer_registry.register("rat")
class RAT(Base):

    def __init__(
        self,
        d_model,
        num_head=8,
        bias=False,
        chunk_size=64,
        ngroups=2,
        init=None,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": kwargs.get("device", "cuda"),
                          "dtype": kwargs.get("dtype", torch.float32)}
        self.layer_id = kwargs.get("layer_id", 0)
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.num_head = num_head
        assert self.d_model % self.num_head == 0
        assert bias is False
        self.d_head = self.d_model // self.num_head
        self.chunk_size = chunk_size
        self.ngroups = ngroups
        self.softmax_scale = self.d_head ** -0.5
        self.in_proj = nn.Linear(d_model, 3 * self.d_model + 2 * ngroups * self.d_head, bias=bias, **factory_kwargs)
        self.gate_bias = nn.Parameter(torch.empty(self.d_model, **factory_kwargs))
        self.input_norm = norm_registry["rmsnorm"](self.d_model, eps=1.0e-6, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bias, **factory_kwargs)
        self.init = init

    def init_weights(self, init_config):
        super().init_weights(init_config)
        if self.chunk_size != 1:
            torch.nn.init.uniform_(self.gate_bias, 1, self.chunk_size - 1)
            self.gate_bias.data = torch.log(self.gate_bias.data)

    def apply_rope(self, q, k, **kwargs):
        rotary_pos_emb = kwargs.get(f"interrope_{self.chunk_size}", None) # (b a l p)
        if rotary_pos_emb is None:
            raise NotImplementedError
        cos, sin = rotary_pos_emb
        cos, sin = cos[None, None, :, :], sin[None, None, :, :]
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)
        return q_rope.to(k.dtype), k_rope.to(k.dtype)

    def prepare_input(self, hidden_states) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inp = self.in_proj(hidden_states)
        z, x, g, q, k = torch.split(inp, [self.d_model, self.d_model, self.d_model, self.ngroups * self.d_head, self.ngroups * self.d_head], dim=-1)
        return torch.sigmoid(z), x, torch.sigmoid((g + self.gate_bias).to(torch.float32)), q, k

    def prepare_output(self, out, z=None): # (out: b l p) 
        out = z * out
        return self.out_proj(out)

    def block_causal_mask(self, b, h, q_idx, kv_idx):
        return q_idx // self.chunk_size > kv_idx

    def forward(self, hidden_states, cache: RATCache=None, **kwargs):
        bs, seq_len, _ = hidden_states.shape
        assert seq_len % self.chunk_size == 0
        num_chunk = seq_len // self.chunk_size
        shortcut = hidden_states
        hidden_states = self.input_norm(hidden_states)
        z, x, g, q, k = self.prepare_input(hidden_states)

        k = k.repeat(1, 1, self.num_head // self.ngroups)
        q = q.repeat(1, 1, self.num_head // self.ngroups).reshape(bs, seq_len, self.num_head, self.d_head).transpose(1, 2)
        x, g, k = map(lambda m: m.reshape(bs, num_chunk, self.chunk_size, self.d_model), (x, g, k))
        g = g.repeat(1, 1, 1, 2) # (b c l d)
        intra_xk = ascan(g, ((1.0 - g) * torch.cat([x, k], dim=-1))).to(torch.bfloat16)
        intra_x, intra_k = intra_xk[..., :self.d_model].reshape(bs, seq_len, self.d_model), intra_xk[..., self.d_model:].reshape(bs, seq_len, self.d_model)
        if cache is not None:
            shift = kwargs.get("seq_end", seq_len)
            cache.cache[self.layer_id][2][cache.bs_start: cache.bs_start + bs].copy_(intra_k[:, shift - 1: shift])
            cache.cache[self.layer_id][3][cache.bs_start: cache.bs_start + bs].copy_(intra_x[:, shift - 1: shift])

        intra_x = intra_x.reshape(bs, seq_len, self.num_head, self.d_head).transpose(1, 2)
        intra_k = intra_k.reshape(bs, seq_len, self.num_head, self.d_head).transpose(1, 2)
        q, intra_k = self.apply_rope(q, intra_k, **kwargs)
        chunk_intra_k = intra_k[..., self.chunk_size - 1: : self.chunk_size, :]
        chunk_intra_x = intra_x[..., self.chunk_size - 1: : self.chunk_size, :]

        if cache is not None:
            cache.cache[self.layer_id][0][cache.bs_start: cache.bs_start + bs, :, cache.chunk_start: cache.chunk_start + num_chunk].copy_(chunk_intra_k)
            cache.cache[self.layer_id][1][cache.bs_start: cache.bs_start + bs, :, cache.chunk_start: cache.chunk_start + num_chunk].copy_(chunk_intra_x)

        block_mask = fla.create_block_mask(self.block_causal_mask, 1, 1, q.shape[2], num_chunk, device="cuda")
        inter_out, inter_lse = fla.flex_attention(q, chunk_intra_k, chunk_intra_x, scale=self.softmax_scale, block_mask=block_mask, return_lse=True)
        intra_lse = (torch.einsum("balp,balp->bal", q, intra_k) * self.softmax_scale).to(torch.float32)
        out = merge_last_token_naive(inter_out, intra_x, inter_lse, intra_lse).transpose(1, 2).reshape(bs, seq_len, self.d_model)
        final_out = self.prepare_output(out, z) + shortcut
        return final_out

    def step(self, hidden_states, cache: RATCache=None, **kwargs): # (b, 1, d)
        shortcut = hidden_states
        bs, seq_len, _ = hidden_states.shape
        kcache, vcache = cache.cache[self.layer_id][0][cache.bs_start: cache.bs_start + bs], cache.cache[self.layer_id][1][cache.bs_start: cache.bs_start + bs]
        lastkcache, lastvcache = cache.cache[self.layer_id][2][cache.bs_start: cache.bs_start + bs], cache.cache[self.layer_id][3][cache.bs_start: cache.bs_start + bs]
        hidden_states = self.input_norm(hidden_states)
        z, x, g, q, k = self.prepare_input(hidden_states)
        k = k.repeat(1, 1, self.num_head // self.ngroups)
        q = q.repeat(1, 1, self.num_head // self.ngroups)
        if cache.seq_start % self.chunk_size == 0:
            new_k = ((1.0 - g) * k).to(torch.bfloat16)
            new_x = ((1.0 - g) * x).to(torch.bfloat16)
        else:
            new_k = (g * lastkcache + (1.0 - g) * k).to(torch.bfloat16)
            new_x = (g * lastvcache + (1.0 - g) * x).to(torch.bfloat16)
        lastkcache.copy_(new_k)
        lastvcache.copy_(new_x)
        q, new_k, new_x = map(lambda m: m.reshape(bs, seq_len, self.num_head, self.d_head).transpose(1, 2), (q, new_k, new_x))
        q, new_k = self.apply_rope(q, new_k, **kwargs)
        kcache[:, :, cache.chunk_start: cache.chunk_start + 1].copy_(new_k)
        vcache[:, :, cache.chunk_start: cache.chunk_start + 1].copy_(new_x)
        if cache.chunk_start == 0:
            attn_out = new_x
        else:
            attn_out = F.scaled_dot_product_attention(q,
                                                      kcache[:, :, :cache.chunk_start + 1],
                                                      vcache[:, :, :cache.chunk_start + 1],
                                                      is_causal=False).transpose(1, 2).reshape(bs, seq_len, self.d_model)
        final_out = self.prepare_output(attn_out, z) + shortcut
        return final_out

    def nflops(self, bs, seq_len):
        pass

    @property
    def nparams(self,):
        return sum([w.numel() for w in self.parameters()])

    @staticmethod
    def get_ckpt_name(model_config):
        return (
            model_config._name_
            + "l"
            + f"{model_config.chunk_size}"
        )