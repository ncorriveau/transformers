import torch
import torch.nn as nn

from .attention import Attention
from .positional_encoding import PositionalEncoding, SinusoidalPE


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


class Skip(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, residual):
        return x + residual


class TransformerBlock(nn.Module):
    def __init__(
        self,
        attention: Attention,
        positional_encoding: PositionalEncoding,
        ffn: FeedForward,
        norm: nn.Module,
        transformer_config: list[str],
    ):
        super().__init__()
        self.attention = attention
        self.positional_encoding = positional_encoding
        self.ffn = ffn
        self.norm = norm
        self.transformer_config = transformer_config
        self.components = nn.ModuleList()
        self.component_types = []
        for component in transformer_config:
            match component:
                case "attention":
                    self.components.append(self.attention)
                case "ffn":
                    self.components.append(self.ffn)
                case "norm":
                    self.components.append(self.norm)
                case "positional_encoding":
                    self.components.append(self.positional_encoding)
                case "skip":
                    self.components.append(Skip())
            self.component_types.append(component)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_input = None
        for component, component_type in zip(self.components, self.component_types):
            if component_type == "skip":
                x = component(x, skip_input)
                skip_input = None
            else:
                if skip_input is None:
                    skip_input = x
                x = component(x)
        return x
