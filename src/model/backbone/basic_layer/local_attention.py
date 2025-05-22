import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import flex_attention as fla
from ....utils.registry import layer_registry, activation_registry, norm_registry, init_registry
from ....utils import config as util_config
from ...base import Base
from ..cache import LocalAttentionCache
from ..util import apply_rotary_pos_emb


@layer_registry.register("local_attention")
class LocalAttention(Base):

    def __init__(
        self,
        d_model,
        num_head=8,
        bias=False,
        init=None,
        window_size=1024,
        ln="rmsnorm",
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {
            "device": kwargs.get("device", "cuda"),
            "dtype": kwargs.get("dtype", torch.float32),
        }
        self.d_model = d_model
        self.num_head = num_head
        self.window_size = window_size
        self.layer_id = kwargs.get("layer_id", 0)
        assert self.d_model % self.num_head == 0
        self.d_head = self.d_model // self.num_head
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=bias, **factory_kwargs)
        self.input_norm = norm_registry[ln](self.d_model, eps=1.0e-6, **factory_kwargs)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.init = init

    def init_weights(self, init_config):
        super().init_weights(init_config)

    def apply_rope(self, q, k, **kwargs): # (b a l p)
        rotary_pos_emb = kwargs.get("rope", None)
        if rotary_pos_emb is None:
            raise NotImplementedError
        cos, sin = rotary_pos_emb
        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos[None, None, :, :], sin[None, None, :, :])
        return q_rope.to(k.dtype), k_rope.to(k.dtype)

    def prepare_input(self, hidden_states):
        xqk = self.in_proj(hidden_states)
        x, q, k = torch.split(xqk, [self.d_model, self.d_model, self.d_model], dim=-1)
        return x, q, k

    def block_causal_mask(self, b, h, q_idx, kv_idx):
        causal_mask = q_idx >= kv_idx
        window_mask = q_idx - kv_idx <= self.window_size
        return causal_mask & window_mask

    def forward(self, hidden_states, cache: LocalAttentionCache=None, **kwargs):
        shortcut = hidden_states
        bs, seq_len, _ = hidden_states.shape # (b l p)
        hidden_states = self.input_norm(hidden_states)
        x, q, k = self.prepare_input(hidden_states)
        x = x.reshape(bs, seq_len, self.num_head, self.d_head).transpose(1, 2)
        q = q.reshape(bs, seq_len, self.num_head, self.d_head).transpose(1, 2)
        k = k.reshape(bs, seq_len, self.num_head, self.d_head).transpose(1, 2)
        q, k = self.apply_rope(q, k, **kwargs)
        if cache is not None: # (b a l p)
            seq_end = kwargs.get("seq_end", seq_len)
            window_shift = seq_end - self.window_size
            k_window, x_window = k[:, :, max(0, window_shift): seq_end], x[:, :, max(0, window_shift): seq_end]
            cache.cache[self.layer_id][0][cache.bs_start: cache.bs_start + bs, :, cache.seq_start: cache.seq_start + k_window.shape[2]].copy_(k_window)
            cache.cache[self.layer_id][1][cache.bs_start: cache.bs_start + bs, :, cache.seq_start: cache.seq_start + x_window.shape[2]].copy_(x_window)
        block_mask = fla.create_block_mask(self.block_causal_mask, 1, 1, q.shape[2], k.shape[2], device="cuda")
        attn_out = fla.flex_attention(q, k, x, block_mask=block_mask).transpose(1, 2).reshape(bs, seq_len, self.d_model)
        final_out = self.out_proj(attn_out) + shortcut
        return final_out

    def step(self, hidden_states, cache: LocalAttentionCache=None, **kwargs): # hidden_states: (b 1 p)
        # cache.seq_start here indicates where we should put the current one 
        shortcut = hidden_states
        bs, seq_len, _ = hidden_states.shape # (b l p)
        assert seq_len == 1
        hidden_states = self.input_norm(hidden_states)
        x, q, k = self.prepare_input(hidden_states)
        x, q, k = map(lambda m: m.reshape(bs, seq_len, self.num_head, self.d_head).transpose(1, 2), (x, q, k))
        q, k = self.apply_rope(q, k, **kwargs)
        kcache, vcache = cache.cache[self.layer_id][0][cache.bs_start: cache.bs_start + bs], cache.cache[self.layer_id][1][cache.bs_start: cache.bs_start + bs]
        kcache[:, :, cache.seq_end: cache.seq_end + 1].copy_(k)
        vcache[:, :, cache.seq_end: cache.seq_end + 1].copy_(x)
        attn_out = F.scaled_dot_product_attention(q,
                                                  kcache[:, :, :cache.seq_end + 1],
                                                  vcache[:, :, :cache.seq_end + 1],
                                                  is_causal=False).transpose(1, 2).reshape(bs, seq_len, self.d_model)
        kcache[:, :, cache.seq_start: cache.seq_start + seq_len].copy_(k)
        vcache[:, :, cache.seq_start: cache.seq_start + seq_len].copy_(x)
        final_out = self.out_proj(attn_out) + shortcut
        return final_out

    def nflops(self, bs, seq_len):
        pass

    @property
    def nparams(self):
        return sum([w.numel() for w in self.parameters()])

    @staticmethod
    def get_ckpt_name(model_config):
        return model_config._name_
