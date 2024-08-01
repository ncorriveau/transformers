import torch
import torch.nn as nn
from attention import Attention
from positional_encoding import PositionalEncoding, SinusoidalPE


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_size: int,
        activation: nn.Module = nn.GELU,
        output_drop: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_size)
        self.activation = activation()
        self.fc2 = nn.Linear(ffn_size, hidden_size)
        self.output_drop = nn.Dropout(output_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.output_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(
        self,
        attention: Attention,
        positional_encoding: PositionalEncoding,
        ffn: FeedForward,
        norm: nn.Module,
        pre_norm: bool = False,
    ):
        self.attention = attention
        self.positional_encoding = positional_encoding
        self.ffn = ffn
        self.norm = norm
        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.attention(self.norm(x))
            x = x + self.ffn(self.norm(x))
        else:
            x = self.norm(x + self.attention(x))
            x = self.norm(x + self.ffn(x))
        return x
