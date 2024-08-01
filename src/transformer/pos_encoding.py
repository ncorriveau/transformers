import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, context_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.context_size = context_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class SinusoidalPE(PositionalEncoding):
    # implementation taken from https://benjaminwarner.dev/2023/07/28/rest-of-the-transformer
    def __init__(self, hidden_size, context_size):
        super().__init__(hidden_size, context_size)
        pe = torch.zeros(context_size, hidden_size)
        position = torch.arange(context_size).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2)
            * -(torch.log(torch.tensor(10000.0)) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
