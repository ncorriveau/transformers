## implement popular attention variants using a generalized framework in PyTorch ##
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

import math
from functools import cached_property

import torch
import torch.nn as nn


class Mask:
    """
    A class to implement many of the attention variants that can be attributed
    to different masking techniques on the attention matrix to reduce computation.
    A good overview of a lot of the methods can be found here:
        https://arxiv.org/abs/2004.05150v2?ref=research.character.ai

    """

    def __init__(self, context_size: int):
        self.context_size = context_size
        self.range = torch.arange(context_size)
        self.diff = self.range[None, :] - self.range[:, None]

    @cached_property
    def causal_mask(self) -> torch.BoolTensor:
        return self.diff <= 1

    def sliding_window_mask(
        self, window_size: int, causal: bool = True
    ) -> torch.BoolTensor:
        mask = self.diff >= -window_size
        if causal:
            mask = mask & self.causal_mask
        return mask.to(torch.bool)

    def global_mask(
        self, h_indices: torch.Tensor, v_indices: torch.Tensor, causal: bool = True
    ) -> torch.BoolTensor:
        if (torch.max(v_indices) >= self.context_size) | (
            torch.max(h_indices) >= self.context_size
        ):
            raise ValueError('Indices cannot exceed context size')

        mask = torch.zeros(self.context_size, self.context_size, dtype=torch.bool)
        mask[:, v_indices] = True
        mask[h_indices, :] = True
        if causal:
            mask[~self.causal_mask] = False
        return mask

    def dilated_sliding_mask(self) -> torch.BoolTensor:
        pass

    def streaming_mask(
        self, anchor_range: int, window_range: int, causal: bool = True
    ) -> torch.BoolTensor:
        """
        Rough implementation of the mask used for KV cache in 'StreamLLM'
        The whole idea centers around 'attention sinks' of the first couple of tokens
        having an outsized impact of attention scores. So keep 'anchor_range' of these,
        and then this would usually be paired with the sliding window mask.

        Paper: https://arxiv.org/abs/2309.17453
        """
        mask = torch.zeros(self.context_size, self.context_size, dtype=torch.bool)
        mask[:, :anchor_range] = True
        mask = mask | self.sliding_window_mask(window_range)
        if causal:
            mask[~self.causal_mask] = False
        return mask


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads_q: int,
        num_heads_k: int,
        num_heads_v: int,
        context_size: int,
        *,
        mask: torch.BoolTensor,
        rotation: torch.Tensor = None,
        attn_drop: float = 0.1,
        output_drop: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        assert (
            hidden_size % num_heads_q == 0
        ), 'Hidden size must be divisible by the number query of heads'
        assert (
            hidden_size % num_heads_k == 0
        ), 'Hidden size must be divisible by the number key of heads'
        assert (
            hidden_size % num_heads_v == 0
        ), 'Hidden size must be divisible by the number of value heads'

        assert (
            num_heads_q % num_heads_k == 0
        ), 'Number of query heads must be divisible by the number of key heads'
        assert (
            num_heads_q % num_heads_v == 0
        ), 'Number of query heads must be divisible by the number of value heads'

        self.num_heads_q = num_heads_q
        self.num_heads_k = num_heads_k
        self.num_heads_v = num_heads_v
        self.rotation = rotation

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
        assert mask.size() == (context_size, context_size), 'Mask size is invalid'
        self.mask = ~mask.view(1, 1, context_size, context_size)

    def forward(self, input: torch.Tensor):
        B, S, D = input.size()  # batch size, sequence length, hidden dim
        Q: torch.Tensor = self.WQ(input)
        K: torch.Tensor = self.WK(input)
        V: torch.Tensor = self.WV(input)
        # add in rotations if PE is RoPE
        if self.rotation:
            Q = Q.reshape(B, S, self.num_heads_q, self.dim_q_k)
            K = K.reshape(B, S, self.num_heads_k, self.dim_q_k)
            Q, K = self.rotation(Q, K)

        Q = Q.reshape(
            B, self.num_heads_q // self.num_heads_k, self.num_heads_k, S, self.dim_q_k
        )
        K = K.reshape(B, self.num_heads_k, S, self.dim_q_k)
        V = V.reshape(B, self.num_heads_v, S, self.dim_v)

        attn: torch.Tensor = (Q @ K.unsqueeze(1).transpose(-2, -1)).reshape(
            B, self.num_heads_q, S, S
        )
        attn = attn / math.sqrt(K.size(-1))

        # take only the mask tokens up to the sequence length
        device = input.device
        self.mask = self.mask.to(device)
        attn = attn.masked_fill(self.mask[:, :, :S, :S], float('-inf'))
        attn = attn.softmax(dim=-1)

        # now we can multiply the attention with the value matrix
        x: torch.Tensor = attn.reshape(
            B, self.num_heads_q // self.num_heads_v, self.num_heads_v, S, S
        ) @ V.unsqueeze(1)
        x = x.reshape(B, S, self.num_heads_q * self.dim_v)
        return self.W_0(x)


if __name__ == '__main__':
    B = 32  # batch size
    S = 10  # sequence length
    H = 768  # hidden size

    hidden_size = 768
    num_heads_q = 12
    num_heads_k = 4
    num_heads_v = 4
    context_size = S

    # simple causal mask below but in theory you could do something like
    # mask = Mask(context_size)
    # combined_mask = mask.causal_mask + mask.sliding_window_mask(4)
    mask = Mask(context_size).causal_mask

    input = torch.rand(B, S, H)
    attention = Attention(
        hidden_size, num_heads_q, num_heads_k, num_heads_v, context_size, mask=mask
    )
    output = attention(input)
    print(output.size())
