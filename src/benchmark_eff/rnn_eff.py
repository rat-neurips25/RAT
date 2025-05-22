import os
import sys
import torch
import math
from triton.testing import do_bench
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.utils.registry import get_all_registries
registries = get_all_registries()
import src.model
import src.task
import src.optim
import src.data
for registry in registries:
    registry._is_register = False

from src.model.backbone.rnn import RNN as rnn_module
from src.model.backbone.cache import RNNCache
from config import *


class RNN:

    def __init__(self, bs, seq_len, config: BaseConfig, dtype=torch.bfloat16):
        self.bs = bs
        self.seq_len = seq_len
        self.config = config.to_dict()
        self.dtype = dtype

    def build(self, return_cache=False):
        rnn = rnn_module(**self.config, seq_len=self.seq_len, layer_id=0).cuda()
        torch._dynamo.reset()
        if return_cache:
            cache = RNNCache(self.bs, self.config.d_model, 1, dtype=self.dtype, device="cuda")
            return torch.compile(rnn), cache
        return torch.compile(rnn)

    def bench_train(self, seed=1005):
        torch.cuda.manual_seed(seed)
        inp = (torch.randn(self.bs, self.seq_len, self.config.d_model) * 0.02).cuda()
        model = self.build()
        def train():
            with torch.amp.autocast("cuda", enabled=True, dtype=self.dtype):
                out = model(inp, cache=None)
            out.mean().backward() # also include backward time
        return do_bench(train)

    @torch.no_grad()
    def bench_context(self, seed=1005):
        torch.cuda.manual_seed(seed)
        inp = (torch.randn(self.bs, self.seq_len, self.config.d_model) * 0.02).cuda()
        model, cache = self.build(return_cache=True)
        model.eval()
        cache.reset_cache()
        @torch.amp.autocast("cuda", enabled=True, dtype=self.dtype)
        def context():
            model(inp, cache=cache)
        return do_bench(context)

    @torch.no_grad()
    def bench_gen(self, seed=1005):
        torch.cuda.manual_seed(seed) # just test the last one, should be enough
        inp = (torch.randn(self.bs, 1, self.config.d_model) * 0.02).cuda()
        model, cache = self.build(return_cache=True)
        model.eval()
        cache.reset_cache()
        cache.set_seq(self.seq_len - 1)
        @torch.amp.autocast("cuda", enabled=True, dtype=self.dtype)
        def gen():
            model.step(inp, cache=cache)
        return do_bench(gen)


if __name__ == "__main__":
    config = Config2()
    seq_list = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
    bs_list = [64, 32, 16, 8, 4, 2, 1]
    for i in range(7):
        print("seq: {}, bs: {}".format(seq_list[i], bs_list[i]))
        model = RNN(bs_list[i], seq_list[i], config)
        train_ms = model.bench_train()
        context_ms = model.bench_context()
        gen_ms = model.bench_gen()
        print("train: {:.2f}, context: {:.2f}, gen: {:.4f}".format(train_ms, context_ms, gen_ms))
    bs_gen_list = [64, 128, 256, 512, 1024, 4096]
    for seq in seq_list:
        result = []
        for bs in bs_gen_list:
            model = RNN(bs, seq, config)
            gen_ms = model.bench_gen()
            result.append(str(round(gen_ms, 2)))
        print(f"seq: {seq}, bs: {'/'.join(result)}")
