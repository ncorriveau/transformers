## implement popular attention variants using a generalized framework in PyTorch ## 
import math

import torch
import torch.nn as nn

"""
Design:
    Vanilla attention variants (used of KV efficiency), the attention mechanism 
    is parameterized by the number of q, k, v heads that will be used 
        MQA: N heads for Query, 1 for K, V 
        GQA: N heads for Query, M for K, V (typically M = N / 8)
        MHA: N heads for Query, K, V
    Attention blocks will also have mask variants, should we should have 
    a Mask type that contains different variants that can be passed into the Attention mechanism. 
        for example, we have causal attention mask, but sliding window attention 
        could also be viewed as a form of mask. 
        Look into global + sliding window attention as well to see if 
        this fits the mask paradigm. 

"""

class Attention(nn.Module):
    def __init__(self,
        hidden_size: int,
        num_heads_q: int,
        num_heads_k: int,
        num_heads_v: int,
        context_size: int,
        mask: torch.BoolTensor,
        attn_drop=0.1,
        output_drop=0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self._num_heads_q = num_heads_q
        self._num_heads_k = num_heads_k
        self._num_heads_v = num_heads_v
        self.context_size = context_size
        self.attn_drop = nn.Dropout(attn_drop)
        self.output_drop = nn.Dropout(output_drop) 

        # TODO: need to add in some checks to make sure that the num of 
        # qkv heads make sense with each other e.g. maybe k must equal v 
        @property
        def num_heads_q(self):
            if hidden_size % self._num_heads_q != 0:
                raise ValueError("Hidden size must be divisible by the number of heads")
            return self._num_heads_q
        
        @property
        def num_heads_k(self):
            if hidden_size % self._num_heads_k != 0:
                raise ValueError("Hidden size must be divisible by the number of heads")
            return self._num_heads_k
        
        @property
        def num_heads_v(self):
            if hidden_size % self._num_heads_v != 0:
                raise ValueError("Hidden size must be divisible by the number of heads")
            return self._num_heads_v
        
        self.WQ = nn.Linear(hidden_size, hidden_size // self.num_heads_q)
        self.WK = nn.Linear(hidden_size, hidden_size // self.num_heads_k)
        self.WV = nn.Linear(hidden_size, hidden_size // self.num_heads_v)
        self.W_0 = nn.Linear(hidden_size, hidden_size)
        
        self.mask = mask

    def forward(self, input: torch.Tensor):
        B, S, C = input.size()
        Q = self.WQ(input).reshape(B, S, self.num_heads_q, C // self.num_heads_q)
        K = self.WK(input).reshape(B, S, self.num_heads_k, C // self.num_heads_k)
        V = self.WV(input).reshape(B, S, self.num_heads_v, C // self.num_heads_v)

        print(f"Query Tensor Size: {Q.size()}")
        print(f"Key Tensor Size: {K.size()}")
        print(f"Value Tensor Size: {V.size()}")

        Q, K, V = Q.transpose(3, 1), K.transpose(3, 1), V.transpose(3, 1)
        
        attn = Q @ K.transpose(-2, -1)
        attn = attn / math.sqrt(K.size(-1))

        # TODO apply our mask here. 
        attn = attn.masked_fill(self.mask, float("-inf"))
        attn = attn.softmax(dim=-1)

        # now we can multiply the attention with the value matrix
        x = attn @ V
        x = x.transpose(1, 2).reshape(B, S, C)
        return self.W_0(x)
