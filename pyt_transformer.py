import math
import torch 
import torch.nn as nn

class SingleHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, bias: bool = True):
        super().__init__()
        self.Wqkv = nn.Linear(hidden_size, (hidden_size//4) * 3, bias=bias)
        self.Wo = nn.Linear(hidden_size//4, hidden_size, bias=bias)
    
    def forward(self, x: torch.Tensor):
        B, S, C = x.size()

        q, k, v = self.Wqkv(x).reshape(B, S, 3, C//4).unbind(dim=2)

        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(k.size(-1))

        x = attn.softmax(dim=-1) @ v

        return self.Wo(x)