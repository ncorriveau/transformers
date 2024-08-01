from dataclasses import dataclass

import torch
import torch.nn as nn

from .transformer.attention import Attention, Mask
from .transformer.positional_encoding import PositionalEncoding
from .transformer.transformer import FeedForward, TransformerBlock

# token embedding: this is of size (vocab_size, hidden_size)
# and is basically a look up table for the tokens and projecting them into the hidden dim

# positional encoding: this is of size (context_size, hidden_size) because it is of the form
# (idx) -> hidden_dim e.g. it takes the tokens index in the total context size and
# maps it to some hidden dimension vector that represents its position in the sequence


# TODO: read in model parameters from config
@dataclass
class Config:
    hidden_size: int = 512
    num_heads_q: int = 8
    num_heads_k: int = 8
    num_heads_v: int = 8
    context_size: int = 512
    ffn_size: int = 2048
    num_layers: int = 6
    vocab_size: int = 10000
    dropout: float = 0.1
    activation: nn.Module = nn.GELU


class CausalLLM:
    def __init__(self, config: Config):
        self.config = config
        self.token_embedding = nn.Embedding(
            self.config.vocab_size, self.config.hidden_size
        )
        self.positional_encoding = PositionalEncoding(
            self.config.hidden_size, self.config.context_size
        )
        self.mask = Mask.causal_mask(self.config.context_size)
        self.attention = Attention(
            self.config.hidden_size,
            self.config.num_heads_q,
            self.config.num_heads_k,
            self.config.num_heads_v,
            self.config.context_size,
            mask=self.mask,
            attn_drop=self.config.dropout,
            output_drop=self.config.dropout,
        )
        self.ffn = FeedForward(
            self.config.hidden_size,
            self.config.ffn_size,
            self.config.activation,
            self.config.dropout,
        )
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.attention,
                    self.positional_encoding,
                    self.ffn,
                    self.norm,
                    pre_norm=False,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.head = nn.Linear(self.config.hidden_size, self.config.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of size batch, seq_len
        B, S = x.size()

        # get the token embeddings -> project into batch, seq_len, hidden_size
        x = self.token_embedding(x)

        # add positional encoding
        x = x + self.positional_encoding(x)

        # maybe an optional drop out here

        # iterate through the transformer blocks
        for block in self.blocks:
            x = block(x)

        return self.head(x)


if __name__ == "__main__":
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("hello world")
    tokens = torch.tensor(tokens).unsqueeze(0)
    print(tokens)
