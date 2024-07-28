## implement popular attention variants using a generalized framework in PyTorch ##
import math
from functools import cached_property

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
    Engines: we will separate out the actual computatational engine for the 
    attention operations into separate modules that can be used with 
    your specific attention variation. Starter ideasa include:
        - FlashAttention
        - Paged Attention 
        - Ring Attention

"""


class Mask:
    """
    A class to implement many of the attention variants that can be attributed
    to different masking techniques on the attention matrix to reduce computation.
    A good overview of a lot of the methods can be found here:
        https://arxiv.org/abs/2004.05150v2?ref=research.character.ai

    """

    def __init__(self, context_size: int):
        self.context_size = context_size

    @cached_property
    def causal_mask(self) -> torch.BoolTensor:
        return torch.tril(
            torch.ones([self.context_size, self.context_size], dtype=torch.bool),
            diagonal=1,
        )

    @cached_property
    def sliding_window_mask(self, window_size: int) -> torch.BoolTensor:
        pass

    @cached_property
    def global_mask(self) -> torch.BoolTensor:
        pass

    @cached_property
    def dilated_sliding_mask(self) -> torch.BoolTensor:
        pass

    @cached_property
    def streaming_mask(self) -> torch.BoolTensor:
        pass


class Attention(nn.Module):
    def __init__(
        self,
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

        assert (
            hidden_size % num_heads_q == 0
        ), "Hidden size must be divisible by the number query of heads"
        assert (
            hidden_size % num_heads_k == 0
        ), "Hidden size must be divisible by the number key of heads"
        assert (
            hidden_size % num_heads_v == 0
        ), "Hidden size must be divisible by the number of value heads"

        assert (
            num_heads_q % num_heads_k == 0
        ), "Number of query heads must be divisible by the number of key heads"
        assert (
            num_heads_q % num_heads_v == 0
        ), "Number of query heads must be divisible by the number of value heads"

        self.num_heads_q = num_heads_q
        self.num_heads_k = num_heads_k
        self.num_heads_v = num_heads_v

        self.dim_q_k = hidden_size // num_heads_q  # shared between Q and K matrices
        self.dim_v = hidden_size // num_heads_v

        self.context_size = context_size
        self.attn_drop = nn.Dropout(attn_drop)
        self.output_drop = nn.Dropout(output_drop)

        # WQ will always be hidden_size -> hidden_size but is shown
        # as hidden_size -> dim_q_k * num_heads_q to emphasize the general pattern
        self.WQ = nn.Linear(hidden_size, self.dim_q_k * num_heads_q)

        self.WK = nn.Linear(hidden_size, self.dim_q_k * num_heads_k)
        self.WV = nn.Linear(hidden_size, self.dim_v * num_heads_v)
        self.W_0 = nn.Linear(num_heads_q * self.dim_v, hidden_size)

        # must be context size, context size for the attention mask
        assert mask.size() == (context_size, context_size), "Mask size is invalid"
        self.mask = mask

    def forward(self, input: torch.Tensor):
        B, S, D = input.size()  # batch size, sequence length, hidden dim

        Q: torch.Tensor = self.WQ(input).reshape(
            B, num_heads_q // num_heads_k, num_heads_k, S, self.dim_q_k
        )
        K: torch.Tensor = self.WK(input).reshape(B, num_heads_k, S, self.dim_q_k)
        V: torch.Tensor = self.WV(input).reshape(B, num_heads_v, S, self.dim_v)

        attn: torch.Tensor = (Q @ K.transpose(-2, -1)).reshape(B, num_heads_q, S, S)
        attn = attn / math.sqrt(K.size(-1))

        # take only the mask tokens up to the sequence length
        attn = attn.masked_fill(self.mask[:, :, :S, :S], float("-inf"))
        attn = attn.softmax(dim=-1)

        # now we can multiply the attention with the value matrix
        x: torch.Tensor = (
            attn.reshape(B, num_heads_q // num_heads_v, num_heads_v, S, S) @ V
        )
        x = x.reshape(B, S, num_heads_q * self.dim_v)
        return self.W_0(x)


if __name__ == "__main__":
    B = 1  # batch size
    S = 10  # sequence length
    H = 768  # hidden size

    hidden_size = 768
    num_heads_q = 12
    num_heads_k = 4
    num_heads_v = 4
    context_size = S
    mask = torch.tril(torch.ones([context_size, context_size], dtype=torch.bool))

    input = torch.rand(B, S, H)
    attention = Attention(
        hidden_size, num_heads_q, num_heads_k, num_heads_v, context_size, mask
    )
    output = attention(input)
    print(output.size())
