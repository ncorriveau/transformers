"""Implementation of the transformer model in PyTorch following https://benjaminwarner.dev/2023/07/01/attention-mechanism"""

import math

import torch
import torch.nn as nn


class SingleHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, bias: bool = True):
        super().__init__()
        self.Wqkv = nn.Linear(hidden_size, (hidden_size // 4) * 3, bias=bias)
        self.Wo = nn.Linear(hidden_size // 4, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor):
        B, S, C = x.size()

        q, k, v = self.Wqkv(x).reshape(B, S, 3, C // 4).unbind(dim=2)

        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(k.size(-1))

        x = attn.softmax(dim=-1) @ v

        return self.Wo(x)


class MultiHeadAttention(nn.Module):
    """
    Implements a bidirectional multi-head attention layer. This is good for
    encoder only models like BERT where you have the input sequence attend
    to pass and future tokens.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        attn_drop=0.1,
        output_drop=0.1,
        bias: bool = True,
    ):
        super().__init__()
        assert (
            hidden_size % num_heads == 0
        ), "Hidden size must be divisible by the number of heads"

        self.nh = num_heads
        super().__init__()

        # we define the number of heads such that nh * head dim = hidden size
        # thus once all heads are combined, we will be producing 3 hidden size weight matrices
        # for exampel if the hidden size size was 12 and nh = 3, then each would be dim BxSx4
        # in the paper they display multi head attention as doing individual projections for each head
        # and then concatenating them together
        self.Wqkv = nn.Linear(hidden_size, (hidden_size * 3), bias=bias)

        # project back to input dim like before
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.output_drop = nn.Dropout(output_drop)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None):
        B, S, C = x.size()

        # for each input seq and batch and head, we are producing 3 matrices, each of size input dim / num heads
        x = self.Wqkv(x).reshape(B, S, 3, self.nh, C // self.nh)

        # ok this is a bit tricky -> so for each batch we have a seq of input vectors
        # for each of these currently we have nh number of heads, and for each of these heads we have 3 matrices (q, k, v)
        # so tranpose 3, 1 we get B, nh, 3, S, C//nh (dim of each head)
        # then unbind (dim=2) to get 3 tensors of shape B, nh, S, C//nh
        # i like to ignore batch size so just look like nh, S, C//nh
        q, k, v = x.transpose(3, 1).unbind(dim=2)

        # calc attention now - since each of these tensors have same dimension
        # we just need to swap the last two in order for the matrix multiplication to work
        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(k.size(-1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # now we can multiply the attention with the value matrix
        x = attn @ v
        # the equivalent to the concatenation step in the paper is to reshape our matrix
        # so that we can apply the final linear layer and project back to in put dim
        x = x.transpose(1, 2).reshape(B, S, C)
        return self.output_drop(self.Wo(x))
