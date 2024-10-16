"""
Implementation of normalizations that may not be found in PyTorch 
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    https://github.com/meta-llama/llama/blob/main/llama/model.py
    """

    def __init__(self, d_model: int, epsilon: float = 1e-8):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(d_model))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        return self.weights * output
