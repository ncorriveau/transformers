from typing import MutableMapping, Optional, Tuple

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


# my first pass for pedagogical purposes
class RoPE(PositionalEncoding):
    def __init__(
        self,
        hidden_size: int,
        context_size: int,
        num_q_k_heads: int,
        base: int = 10_000,
    ):
        super().__init__(hidden_size, context_size)
        self.thetas = torch.float_power(
            base, -2 * torch.arange(0, hidden_size // 2) / hidden_size
        )
        self.num_heads = num_q_k_heads
        self.build_rope_cache()

    def build_rope_cache(self) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        # this is our i's
        seq_idx = torch.arange(self.context_size)

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
        batch_size, seq_len, num_heads, head_dim = x.shape

        # extract the values based on whether input_pos is set or not
        rope_cache = self.cache[input_pos] if input_pos else self.cache[:seq_len]

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_shaped = x.float().reshape(batch_size, seq_len, num_heads, head_dim // 2, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(1, seq_len, 1, head_dim // 2, 2)

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
        x_out = x_out.reshape(*x.shape)

        return x_out.type_as(x)


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


class RotaryEmbedding(nn.Module):
    """
    Taken from Olmo Implementation: https://github.com/allenai/OLMo/blob/main/olmo/model.py
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(
        self,
        hidden_size: int,
        context_size: int,
        num_q_k_heads: int,
        cache: BufferCache = BufferCache(),
        base: int = 10_000,
    ):
        super().__init__()
        self.__cache = cache
        self.hidden_size = hidden_size
        self.context_size = context_size
        self.num_q_k_heads = num_q_k_heads
        self.base = base
        # Warm up cache.
        self.get_rotary_embedding(context_size, torch.device("cpu"))

    def get_rotary_embedding(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[1] >= seq_len
            and pos_cos.shape[1] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :seq_len, :, :], pos_cos[:, :seq_len, :, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.hidden_size
            inv_freq = 1.0 / (
                self.base
                ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = (
                positions.sin()[None, :, None, :],
                positions.cos()[None, :, None, :],
            )
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """ "
        This is rotating the input vector by 90 degrees. Once we do this,
        Then simplying multiplying by sin(theta) and adding to cos(theta) * x
        will be the same as applying a rotational matrix [cos(theta), -sin(theta); sin(theta), cos(theta)] to x
        """
        B, T, nh, hs = x.size()
        x = x.view(B, T, nh, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(
        self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_, k_ = q.float(), k.float()
        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = (
                q_.shape[1],
                k_.shape[1],
            )  # could be different if layer_past not None
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, key_len - query_len : key_len, :, :],
                pos_cos[:, key_len - query_len : key_len, :, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)


if __name__ == "__main__":
    hidden_size = 2
    context_size = 3
    batch_size = 1
    num_heads = 1
    x1 = torch.rand(batch_size, context_size, num_heads, hidden_size)
    x2 = torch.rand(batch_size, context_size, num_heads, hidden_size)

    cache = BufferCache()
    pe2 = RotaryEmbedding(hidden_size, context_size, num_heads, cache, base=10)
    actual_result = pe2(x1, x2)
    print(f"original x1: {x1.squeeze()},\n rotated: {actual_result[0].squeeze()}")
    print(f"original x2: {x2.squeeze()},\n rotated: {actual_result[1].squeeze()}")

    # Plotting the vectors
    def plot_vectors(original, rotated, title):
        import matplotlib.pyplot as plt

        plt.figure()
        plt.quiver(
            0,
            0,
            original[0],
            original[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="r",
            label="Original",
        )
        plt.quiver(
            0,
            0,
            rotated[0],
            rotated[1],
            angles="xy",
            scale_units="xy",
            scale=1,
            color="b",
            label="Rotated",
        )
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.axhline(0, color="grey", lw=0.5)
        plt.axvline(0, color="grey", lw=0.5)
        plt.grid()
        plt.legend()
        plt.title(title)
        plt.show()

    # Extracting the vectors
    original_x1 = x1.squeeze().numpy()[1]
    rotated_x1 = actual_result[0].squeeze().detach().numpy()[1]
    original_x2 = x2.squeeze().numpy()[1]
    rotated_x2 = actual_result[1].squeeze().detach().numpy()[1]

    # Plot the vectors
    plot_vectors(original_x1, rotated_x1, "Vector x1 Before and After Rotation")
    plot_vectors(original_x2, rotated_x2, "Vector x2 Before and After Rotation")
