import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from ...utils.registry import norm_registry, backbone_registry, layer_registry
from ...utils import config as util_config
from ..base import Base


class SequenceBlock(nn.Module):

    def __init__(
        self,
        seq_cell,
        hidden_cell,
        **kwargs,
    ):
        super().__init__()
        self.seq_layer = util_config.instantiate(layer_registry, seq_cell, **kwargs)
        self.hidden_layer = util_config.instantiate(layer_registry, hidden_cell, **kwargs)

    def forward(self, hidden_states, cache=None, **kwargs):
        hidden_states = self.seq_layer(hidden_states, cache, **kwargs)
        hidden_states = self.hidden_layer(hidden_states)
        return hidden_states

    def step(self, hidden_states, cache=None, **kwargs):
        hidden_states = self.seq_layer.step(hidden_states, cache, **kwargs)
        hidden_states = self.hidden_layer.step(hidden_states)
        return hidden_states


@backbone_registry.register("sequence_interleave")
class SequenceInterleaveBackbone(Base):

    def __init__(
        self,
        num_layers,
        d_model,
        seq_cell: dict = None,
        seq_cell1: dict=None,
        hidden_cell: dict = None,
        interleave_step: int = 2,
        ln: str = "layernorm",
        bias: bool = False,
        dropout = 0.0,
        init: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.init = init
        self.interleave_step = interleave_step
        self.layers = []
        num_seq_cell1 = 0
        num_seq_cell = 0
        for i in range(num_layers):
            if i % interleave_step == 0:
                self.layers.append(SequenceBlock(seq_cell1, hidden_cell, layer_id=num_seq_cell1, **kwargs))
                num_seq_cell1 += 1
            else:
                self.layers.append(SequenceBlock(seq_cell, hidden_cell, layer_id=num_seq_cell, **kwargs))
                num_seq_cell += 1
        self.layers = nn.ModuleList(self.layers)
        self.ln = norm_registry[ln](d_model, eps=1.0e-6)
        self.init_weights(init)

    def init_weights(self, init_config):
        for l in self.layers:
            l.seq_layer.init_weights(init_config)
            l.hidden_layer.init_weights(init_config)
        self._init_weights(self.ln)

    def forward(self, hidden_states, cache=None, **kwargs):
        for i, layer in enumerate(self.layers):
            if i % self.interleave_step == 0:
                hidden_states = layer(hidden_states, cache[1] if cache is not None else None, **kwargs)
            else:
                hidden_states = layer(hidden_states, cache[0] if cache is not None else None, **kwargs)
        hidden_states = self.ln(hidden_states)
        return hidden_states

    def step(self, hidden_states, cache=None, **kwargs):
        for i, layer in enumerate(self.layers):
            if i % self.interleave_step == 0:
                hidden_states = layer.step(hidden_states, cache[1], **kwargs)
            else:
                hidden_states = layer.step(hidden_states, cache[0], **kwargs)
        hidden_states = self.ln(hidden_states)
        return hidden_states

    def get_fsdp_policy(self, ): # fsdp1
        return {SequenceBlock}

    @property
    def d_input(self):
        return self.d_model

    @property
    def d_output(self):
        return self.d_model

    @staticmethod
    def get_ckpt_name(model_config):
        return (
            model_config._name_
            + "d"
            + f"{model_config.d_model}"
            + "l"
            + f"{model_config.num_layers}"
            + "-"
            + layer_registry.get(model_config.seq_cell._name_).get_ckpt_name(model_config.seq_cell)
            + "-"
            + layer_registry.get(model_config.seq_cell1._name_).get_ckpt_name(model_config.seq_cell1)
            + "-"
            + layer_registry.get(model_config.hidden_cell._name_).get_ckpt_name(model_config.hidden_cell)
        )

    def nflops(self, bs, seq_len):
        raise NotImplementedError
    
    @property
    def nparams(self):
        raise NotImplementedError
