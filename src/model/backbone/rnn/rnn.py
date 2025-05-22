import torch
import torch.nn as nn
from typing import Tuple
import torch.nn.functional as F
from ....utils.registry import layer_registry, init_registry, activation_registry, norm_registry
from ....utils import config as util_config
from ...base import Base
from ...op import pscan, ascan
from ..cache import RNNCache


@layer_registry.register("rnn")
class RNN(Base):

    def __init__(
        self,
        d_model,
        bias=False,
        seq_len=1024,
        init=None,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": kwargs.get("device", "cuda"),
                          "dtype": kwargs.get("dtype", torch.float32)}
        self.d_model = d_model
        assert bias is False
        self.in_proj = nn.Linear(d_model, 3 * self.d_model, bias=bias, **factory_kwargs)
        self.gate_bias = nn.Parameter(torch.empty(self.d_model, **factory_kwargs)) # for the forget gate
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=bias, **factory_kwargs)
        self.input_norm = norm_registry["rmsnorm"](self.d_model, eps=1.0e-6, **factory_kwargs)
        self.init = init
        self.seq_len = seq_len
        self.layer_id = kwargs.get("layer_id", 0)

    def init_weights(self, init_config):
        super().init_weights(init_config)
        torch.nn.init.uniform_(self.gate_bias, 1, self.seq_len - 1)
        self.gate_bias.data = torch.log(self.gate_bias.data)

    def prepare_input(self, hidden_states) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        zxg = self.in_proj(hidden_states)
        z, x, g = torch.split(zxg, [self.d_model, self.d_model, self.d_model], dim=-1)
        return torch.sigmoid(z), x, torch.sigmoid((g + self.gate_bias).to(torch.float32))

    def prepare_output(self, out, z=None): # (out: b l p) 
        out = z * out
        return self.out_proj(out)

    def forward(self, hidden_states, cache: RNNCache=None, **kwargs):
        shortcut = hidden_states
        bs, seq_len, _ = hidden_states.shape
        hidden_states = self.input_norm(hidden_states)
        z, x, g = self.prepare_input(hidden_states) # (b l d)
        new_x = ascan(g.unsqueeze(1), ((1.0 - g) * x).unsqueeze(1)).squeeze(1).to(torch.bfloat16)
        if cache is not None: # (b, 1, d) 
            shift = kwargs.get("seq_end", seq_len)
            cache.cache[self.layer_id][cache.bs_start: cache.bs_start + bs].copy_(new_x[:, shift - 1: shift, :])
        final_out = self.prepare_output(new_x, z) + shortcut
        return final_out

    def step(self, hidden_states, cache: RNNCache=None, **kwargs): # (b l p) cache # (b l p)
        shortcut = hidden_states
        bs = hidden_states.shape[0]
        hidden_states = self.input_norm(hidden_states)
        z, x, g = self.prepare_input(hidden_states)
        rnncache = cache.cache[self.layer_id][cache.bs_start: cache.bs_start + bs]
        if cache.seq_start == 0:
            new_x = ((1.0 - g) * x).to(torch.bfloat16)
        else:
            new_x = (g * rnncache + (1.0 - g) * x).to(torch.bfloat16)
        rnncache.copy_(new_x)
        final_out = self.prepare_output(new_x, z) + shortcut
        return final_out

    def nflops(self, bs, seq_len):
        pass

    @property
    def nparams(self,):
        return sum([w.numel() for w in self.parameters()])

    @staticmethod
    def get_ckpt_name(model_config):
        return model_config._name_