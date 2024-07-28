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

        # mask our padding tokens since we will add them to make every sequence length S
        attn = attn.masked_fill(mask.view(B, 1, 1, S), float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # now we can multiply the attention with the value matrix
        x = attn @ v
        # the equivalent to the concatenation step in the paper is to reshape our matrix
        # so that we can apply the final linear layer and project back to in put dim
        x = x.transpose(1, 2).reshape(B, S, C)
        return self.output_drop(self.Wo(x))


class CausalSelfAttention(nn.Module):
    """
    Implements a causal self attention w multi-head attention layer. This is good for
    language generation where the model should only attend to previous tokens.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        context_size: int,
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

        # use register buffer when you want to add weights that are not updated during backprop
        # good discussion here: https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones([context_size, context_size], dtype=torch.bool), diagonal=1
            ),
        ).view(
            1, 1, context_size, context_size
        )  # this just expands to 4 dim

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None):
        B, S, C = x.size()
        x = self.Wqkv(x).reshape(B, S, 3, self.nh, C // self.nh)
        q, k, v = x.transpose(3, 1).unbind(dim=2)
        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(k.size(-1))

        # apply causal attention mask + our padding mask together
        # this works because we have boolean logic at each place e.g. 1 + 0 = 1
        mask = self.causal_mask[:, :, :S, :S] + mask.view(B, 1, 1, S)
        attn = attn.masked_fill(mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # now we can multiply the attention with the value matrix
        x = attn @ v
        # the equivalent to the concatenation step in the paper is to reshape our matrix
        # so that we can apply the final linear layer and project back to in put dim
        x = x.transpose(1, 2).reshape(B, S, C)
        return self.output_drop(self.Wo(x))


class CausalCrossAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        context_size: int,
        attn_drop=0.1,
        output_drop=0.1,
        bias: bool = True,
    ):
        """
        Following the cross attention implementation of the original transformer paper
        we have one set of query weights for the input seq X (think english input)
        and another set of key and value weights for the other input seq Y (think french output in a translation task)
        """
        assert (
            hidden_size % num_heads == 0
        ), "Hidden size must be divisible by the number of heads"

        self.nh = num_heads
        super().__init__()
        self.Wq = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.Wkv = nn.Linear(hidden_size, hidden_size * 2, bias=bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.output_drop = nn.Dropout(output_drop)

    def forward(self, x: torch.Tensor, y: torch.Tensor, mask: torch.BoolTensor = None):
        # for ex in transformer paper, C = 512, nh = 8, so each head has output dim 64
        B, S, C = x.size()
        q = self.Wq(x).reshape(B, S, self.nh, C // self.nh).transpose(1, 2)

        # add in dimension for the separate k and v weights
        y = self.Wkv(y).reshape(B, S, 2, self.nh, C // self.nh)
        k, v = y.transpose(3, 1).unbind(dim=2)

        # rest is the same as regular self causal attention
        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(k.size(-1))
        mask = self.causal_mask[:, :, :S, :S] + mask.view(B, 1, 1, S)
        attn = attn.masked_fill(mask, float("-inf"))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, S, C)
        return self.output_drop(self.Wo(x))


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expanded_size: int,
        activation: nn.Module = nn.GELU,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, expanded_size, bias=bias)
        self.act = activation()
        self.fc2 = nn.Linear(expanded_size, hidden_size, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.act(self.fc1(x))
        x = self.drop(self.fc2(x))
        return x


class TransformerBlock(nn.Module):
    """
    We are using pre-norm by default
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        context_size: int,
        attention: nn.Module = CausalSelfAttention,
        activation: nn.Module = nn.GELU,
        attn_drop: float = 0.1,
        output_drop: float = 0.1,
        ff_expansion: int = 4,
        ff_drop: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = attention(
            hidden_size, num_heads, context_size, attn_drop, output_drop, bias=bias
        )
        self.ff = FeedForward(
            hidden_size=hidden_size,
            expanded_size=ff_expansion,
            activation=activation,
            dropout=ff_drop,
            bias=bias,
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor):
        # normalization before passed to attention / FFN
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class PositionalEncoding(nn.Module):
    """
    Implements the sinusoidal positional encoding from the original transformer paper
    """

    def __init__(self, hidden_size: int, context_length: int):
        pe = torch.zeros(context_length, hidden_size, dtype=torch.float32)
        pos = torch.arange(context_length).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0) / hidden_size)
        )

        # even use sin, odd use cos
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor):
        # so we have calculated this value for every position up to
        # the max length. now we are taking the input sequence length and only
        # returning values up to there
        return self.pe[:, : x.shape[1], :]


# time to add the whole GPT2 model
class GPT2(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        context_size: int,
        num_layers: int,
        vocab_size: int,
        embed_dropout: float,
        head_norm: bool = True,
        tie_weights: bool = False,
        attn_drop: float = 0.1,
        output_drop: float = 0.1,
        ff_expansion: int = 4,
        ff_drop: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()

        # token emb need to be length of the vocab size
        # while pos emb needs to be length of the context size
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(context_size, hidden_size)
        self.embed_drop = nn.Dropout(embed_dropout)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_heads,
                    context_size,
                    attention=CausalSelfAttention,
                    activation=nn.GELU,
                    attn_drop=attn_drop,
                    output_drop=output_drop,
                    ff_expansion=ff_expansion,
                    ff_drop=ff_drop,
                    bias=bias,
                )
                for _ in range(num_layers)
            ]
        )
        if head_norm:
            self.head_norm = nn.LayerNorm(hidden_size)
        else:
            self.head_norm = nn.Identity()

        if tie_weights:
            self.head_weight = self.token_emb.weight

        self.head = nn.Linear(hidden_size, vocab_size, bias=bias)
        pe = torch.zeros(context_size, hidden_size, dtype=torch.float32)
        self.register_buffer("pe", pe, persistent=False)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor):
        tokens = self.token_emb(x)
        pos = self.pos_emb(self.pe[: x.shape[1]])
        x = self.embed_drop(tokens + pos)

        for layer in self.layers:
            x = layer(x)

        x = self.head_norm(x)
        return self.head(x)