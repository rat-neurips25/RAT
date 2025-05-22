import torch


class AttentionCache: # k after rope

    def __init__(self, 
                 max_bs,
                 max_seq_len,
                 num_head,
                 d_head,
                 num_layer=1,
                 dtype=torch.bfloat16,
                 device="cuda"):
        self.max_bs = max_bs
        self.max_seq_len = max_seq_len
        self.num_layer = num_layer
        self.num_head = num_head
        self.d_head = d_head
        self.cache = dict()
        self.bs_start = 0
        self.seq_start = 0
        for i in range(num_layer):
            kcache = torch.empty(self.max_bs, self.num_head, self.max_seq_len, \
                                     self.d_head, dtype=dtype, device=device)
            vcache = torch.empty(self.max_bs, self.num_head, self.max_seq_len, \
                                     self.d_head, dtype=dtype, device=device)
            self.cache[i] = (kcache, vcache)

    def set_seq(self, i):
        self.seq_start = i

    def reset_cache(self, ):
        self.seq_start = 0
        self.bs_start = 0


class LocalAttentionCache: # k after rope

    def __init__(self, 
                 max_bs,
                 window_size, # plus 1 to hold the current token
                 num_head,
                 d_head,
                 num_layer=1,
                 dtype=torch.bfloat16,
                 device="cuda"):
        self.max_bs = max_bs
        self.window_size = window_size
        self.num_layer = num_layer
        self.num_head = num_head
        self.d_head = d_head
        self.cache = dict()
        self.bs_start = 0
        self.seq_start = 0
        self.seq_end = 0
        for i in range(num_layer):
            kcache = torch.empty(self.max_bs, self.num_head, self.window_size + 1, \
                                     self.d_head, dtype=dtype, device=device)
            vcache = torch.empty(self.max_bs, self.num_head, self.window_size + 1, \
                                     self.d_head, dtype=dtype, device=device)
            self.cache[i] = (kcache, vcache)

    def set_seq(self, i):
        self.seq_end = min(self.window_size, i) # the position to temporarily store the current token
        self.seq_start = (self.seq_start + 1) % self.window_size

    def reset_cache(self, ):
        self.seq_start = 0
        self.seq_end = 0
        self.bs_start = 0

class RNNCache:

    def __init__(self,
                 max_bs,
                 d_model,
                 num_layer=1,
                 dtype=torch.bfloat16,
                 device="cuda"):
        self.max_bs = max_bs
        self.d_model = d_model
        self.bs_start = 0
        self.seq_start = 0
        self.cache = dict()
        for i in range(num_layer):
            self.cache[i] = torch.zeros(self.max_bs, 1, self.d_model, dtype=dtype, device=device)

    def set_seq(self, i):
        self.seq_start = i

    def reset_cache(self, ):
        self.seq_start = 0
        self.bs_start = 0


class RATCache:

    def __init__(self,
                 max_bs,
                 max_num_chunk,
                 chunk_size,
                 num_head,
                 d_head,
                 d_model,
                 num_layer=1,
                 dtype=torch.bfloat16,
                 device="cuda"):
        self.max_bs = max_bs
        self.max_num_chunk = max_num_chunk
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.num_head = num_head
        self.d_head = d_head
        self.cache = dict()
        self.bs_start = 0
        self.chunk_start = 0
        self.seq_start = 0
        for i in range(num_layer):
            # ensure each chunk starts from 0
            kcache = torch.empty(self.max_bs, self.num_head, self.max_num_chunk, self.d_head, dtype=dtype, device=device)
            vcache = torch.empty(self.max_bs, self.num_head, self.max_num_chunk, self.d_head, dtype=dtype, device=device)
            lastkcache = torch.zeros(self.max_bs, 1, self.d_model, dtype=dtype, device=device)
            lastvcache = torch.zeros(self.max_bs, 1, self.d_model, dtype=dtype, device=device)
            self.cache[i] = (kcache, vcache, lastkcache, lastvcache)

    def set_seq(self, i):
        self.seq_start = i
        self.chunk_start = i // self.chunk_size

    def reset_cache(self, ):
        self.seq_start = 0
        self.chunk_start = 0
        self.bs_start = 0