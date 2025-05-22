import os
import sys
import torch
from transformers import LlamaTokenizer
from triton.testing import do_bench
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.utils.registry import get_all_registries
registries = get_all_registries()
import src.task
import src.optim
import src.data
for registry in registries:
    registry._is_register = False

from src.model.backbone.basic_layer.attention import Attention as attention_module
from src.model.embedding.pe import RoPE
from src.model.backbone.cache import AttentionCache
from config import *


class Attention:

    def __init__(self, bs, seq_len, config: BaseConfig, dtype=torch.bfloat16):
        self.bs = bs
        self.seq_len = seq_len
        self.config = config.to_dict()
        self.dtype = dtype

    def build(self, return_cache=False):
        attention = attention_module(**self.config, layer_id=0).cuda()
        rope = RoPE(dim=self.config.d_head, max_seq_len=self.seq_len, base=10000, device="cuda")
        torch._dynamo.reset()
        if return_cache:
            cache = AttentionCache(self.bs, self.seq_len, self.config.num_head, self.config.d_head, 1, dtype=self.dtype, device="cuda")
            return torch.compile(attention), rope, cache
        return torch.compile(attention), rope

    def bench_train(self, seed=1005):
        torch.cuda.manual_seed(seed)
        inp = (torch.randn(self.bs, self.seq_len, self.config.d_model) * 0.02).cuda()
        model, rope = self.build()
        _, rope_kwargs = rope(0, self.seq_len, "cuda", self.dtype)
        def train():
            with torch.amp.autocast("cuda", enabled=True, dtype=self.dtype):
                out = model(inp, cache=None, **rope_kwargs)
            out.mean().backward() # also include backward time
        return do_bench(train)

    @torch.no_grad()
    def bench_context(self, seed=1005):
        torch.cuda.manual_seed(seed)
        inp = (torch.randn(self.bs, self.seq_len, self.config.d_model) * 0.02).cuda()
        model, rope, cache = self.build(return_cache=True)
        model.eval()
        cache.reset_cache()
        _, rope_kwargs = rope(0, self.seq_len, "cuda", self.dtype)
        @torch.amp.autocast("cuda", enabled=True, dtype=self.dtype)
        def context():
            model(inp, cache=cache, **rope_kwargs)
        return do_bench(context)

    @torch.no_grad()
    def bench_gen(self, seed=1005):
        torch.cuda.manual_seed(seed) # just test the last one, should be enough
        inp = (torch.randn(self.bs, 1, self.config.d_model) * 0.02).cuda()
        model, rope, cache = self.build(return_cache=True)
        model.eval()
        cache.reset_cache()
        cache.set_seq(self.seq_len - 1)
        _, rope_kwargs = rope.step(self.seq_len - 1, self.seq_len, "cuda", self.dtype)
        @torch.amp.autocast("cuda", enabled=True, dtype=self.dtype)
        def gen():
            model.step(inp, cache=cache, **rope_kwargs)
        return do_bench(gen)


if __name__ == "__main__":
    config = Config2()
    seq_list = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
    bs_list = [64, 32, 16, 8, 4, 2, 1]
    for i in range(7):
        print("seq: {}, bs: {}".format(seq_list[i], bs_list[i]))
        model = Attention(bs_list[i], seq_list[i], config)
        train_ms = model.bench_train()
        context_ms = model.bench_context()
        gen_ms = model.bench_gen()
        print("train: {:.2f}, context: {:.2f}, gen: {:.4f}".format(train_ms, context_ms, gen_ms))

    bs_gen_list = [64, 128, 256, 512, 1024, 4096]
    for seq in seq_list:
        result = []
        for bs in bs_gen_list:
            model = Attention(bs, seq, config)
            try:
                gen_ms = model.bench_gen()
                result.append(str(round(gen_ms, 2)))
            except Exception as e:
                result.append("OOM")
        print(f"seq: {seq}, bs: {'/'.join(result)}")
