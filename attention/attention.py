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

        assert hidden_size % num_heads_q == 0, "Hidden size must be divisible by the number query of heads"
        assert hidden_size % num_heads_k == 0, "Hidden size must be divisible by the number key of heads"
        assert hidden_size % num_heads_v == 0, "Hidden size must be divisible by the number of value heads"

        self.num_heads_q = num_heads_q
        self.num_heads_k = num_heads_k
        self.num_heads_v = num_heads_v

        self.dim_q = hidden_size // num_heads_q
        self.dim_k = hidden_size // num_heads_k
        self.dim_v = hidden_size // num_heads_v

        self.context_size = context_size
        self.attn_drop = nn.Dropout(attn_drop)
        self.output_drop = nn.Dropout(output_drop) 
        
        # TODO: revert this bck to use the verified properties
        self.WQ = nn.Linear(hidden_size, hidden_size)
        self.WK = nn.Linear(hidden_size, hidden_size)
        self.WV = nn.Linear(hidden_size, hidden_size)
        self.W_0 = nn.Linear(hidden_size, hidden_size)

        # TODO: add in checks around the sizing here (must be B, H, S, S)
        self.mask = mask

    def forward(self, input: torch.Tensor):
        B, S, D = input.size()   # batch size, sequence length, hidden dim
        
        Q = self.WQ(input).reshape(B, num_heads_q, S, D // self.num_heads_q)
        K = self.WK(input).reshape(B, num_heads_k, S, D // self.num_heads_k)
        V = self.WV(input).reshape(B, num_heads_v, S, D // self.num_heads_v)
        
        attn = Q @ K.transpose(-2, -1)
        attn = attn / math.sqrt(K.size(-1))

        # TODO apply our mask here. 
        # attn = attn.masked_fill(self.mask, float("-inf"))
        # attn = attn.softmax(dim=-1)

        # now we can multiply the attention with the value matrix
        x = attn @ V
        x = x.transpose(1, 2).reshape(B, S, H)
        return self.W_0(x)


if __name__ == "__main__":
    B = 1    # batch size 
    S = 10   # sequence length
    H = 768  # hidden size 

    hidden_size = 768
    num_heads_q = 12
    num_heads_k = 1
    num_heads_v = 1
    context_size = S
    mask = torch.tril(
        torch.ones([context_size, context_size], 
                                 dtype=torch.bool)
                                 ).view(1, 1, context_size, context_size)

    input = torch.rand(B, S, H)
    attention = Attention(hidden_size, num_heads_q, num_heads_k, num_heads_v, context_size, mask)
    output = attention(input)
    print(output.size())