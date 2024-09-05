import numpy as np
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
    def __init__(self, hidden_size: int, context_size: int):
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


class RoPE(PositionalEncoding):
    def __init__(self, hidden_size: int, context_size: int, base: int = 10_000, m=1):
        super().__init__(hidden_size, context_size)
        self.thetas = (
            np.power(base, -2 * np.arange(0, hidden_size // 2) / hidden_size) * m
        )
        rope = torch.zeros(hidden_size, hidden_size)
        for i in range(len(self.thetas)):
            idx = i * 2
            rope[idx, idx] = np.cos(self.thetas[i])
            rope[idx, idx + 1] = -np.sin(self.thetas[i])
            rope[idx + 1, idx] = np.sin(self.thetas[i])
            rope[idx + 1, idx + 1] = np.cos(self.thetas[i])

        self.register_buffer("rope", rope, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.rope


if __name__ == "__main__":
    hidden_size = 4
    context_size = 1024
    x = torch.rand(1, context_size, hidden_size)
    # pe = SinusoidalPE(hidden_size, context_size)
    # print(pe(x).size())
    pe = RoPE(hidden_size, context_size)
    print(pe(x).size())
