from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# from torchtune.modules.position_embeddings import RotaryPositionalEmbeddings


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
    def __init__(self, hidden_size: int, context_size: int, base: int = 10_000):
        super().__init__(hidden_size, context_size)
        self.thetas = torch.float_power(
            base, -2 * torch.arange(0, hidden_size // 2) / hidden_size
        )
        self.build_rope_cache()

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        # this is our i's
        seq_idx = torch.arange(max_seq_len)

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        # for ex here we have 2 thetas for hidden dim = 4
        # so we produce a tensor of shape [max_seq_len, 2]
        idx_theta = torch.outer(seq_idx, self.thetas).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        # so here we just calculate cos, sin for all of our thetas thus the dim 2 = cos, sin and dim // 2 = thetas
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, input_pos: Optional[int] = None) -> torch.Tensor:
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = self.cache[input_pos] if input_pos else self.cache[:seq_len]

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_shaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, x_shaped.size(1), 1, x_shaped.size(3), 2)

        # so x is now chopped up with each feature split into hidden_dim //2  '2-D coordinates'
        # we then rotate each 2d coordinate by the respective theta_i (cos or sin) for each feature vector
        # the theta_i's are already calculated for hd//2 size and for each m corresponding to the position in the seq
        # so the output will be same shape, just with each 2d coordinate rotated for each feature vector corresponding
        # to its position in the sequence
        x_out = torch.stack(
            [
                x_shaped[..., 0] * rope_cache[..., 0]
                - x_shaped[..., 1] * rope_cache[..., 1],
                x_shaped[..., 1] * rope_cache[..., 0]
                + x_shaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        # x_out tensor has shape [b, s, n_h, h_d // 2, 2]
        # re-collapse on the last dimension to reshape back into h_d
        # final tensor has shape [b, s, n_h, h_d]
        x_out = x_out.reshape(*x_out.shape[:-2], -1)

        return x_out.type_as(x)


if __name__ == "__main__":
    hidden_size = 4
    context_size = 1024
    x = torch.rand(1, context_size, 4, hidden_size)
    pe1 = RoPE(hidden_size, context_size)
