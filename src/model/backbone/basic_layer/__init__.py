from .mlp import FFN
from .attention import Attention
from .local_attention import LocalAttention
from ...base import Identity
from ....utils.registry import layer_registry
layer_registry.register("identity", Identity)