from dataclasses import dataclass
import math
import torch
import torch.nn as nn 
from torch.nn import function as F 

@dataclass
class GPTConfig:
    vocab_size: int = 50257  # we can override this to 50304 which is the closest nice factor of 2 for implementation
    block_size: int = 1024   # this is the context size
    n_layers: int = 12
    n_heads: int = 12
    n_embed: int = 768 

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert (
            config.n_embed % config.n_heads == 0
        ), "Hidden size must be divisible by the number of heads"

        self.n_head = config.n_heads
        self.n_embd = config.n_embed

        # we define the number of heads such that nh * head dim = hidden sizei
        # thus once all heads are combined, we will be producing 3 hidden size weight matrices
        # for exampel if the hidden size size was 12 and nh = 3, then each would be dim BxSx4
        # in the paper they display multi head attention as doing individual projections for each head
        # and then concatenating them together
        self.c_attn = nn.Linear(self.n_embd, self.n_embd * 3)

        # project back to input dim like before
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)

        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones([config.block_size, config.block_size], dtype=torch.bool),
            ),
        ).view(
            1, 1, config.block_size, config.block_size
        )  # this just expands to 4 dim

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor = None):
        B, S, C = x.size()
        x = self.c_attn(x).reshape(B, S, 3, self.nh, C // self.nh)
        q, k, v = x.transpose(3, 1).unbind(dim=2)
        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(k.size(-1))

        # apply causal attention mask + our padding mask together
        # this works because we have boolean logic at each place e.g. 1 + 0 = 1
        mask = self.bias[:, :, :S, :S] + mask.view(B, 1, 1, S)
        attn = attn.masked_fill(mask, float("-inf"))

        attn = attn.softmax(dim=-1)

        # now we can multiply the attention with the value matrix
        x = attn @ v
        # the equivalent to the concatenation step in the paper is to reshape our matrix
        # so that we can apply the final linear layer and project back to in put dim
        x = x.transpose(1, 2).reshape(B, S, C)
        return self.c_proj(x)


class MLP(nn.module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')  # using approx method to replicate paper exactly 
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
    
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)   # tokens communicate here, think of it as a 'reduce' operation (weighted sum across 1024 tokens)
        self.mlp = MLP(config)    # no communication here, we are 'mapping' in this step 

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),   # take total vocab size from tokens and project into embeddings 
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embed)
          )
        )
        self.ln_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size, "Cannot forward, model block size is exhausted"
         