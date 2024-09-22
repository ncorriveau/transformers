from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer.attention import Attention, Mask
from .transformer.positional_encoding import PositionalEncoding, RoPE, SinusoidalPE
from .transformer.transformer import FeedForward, TransformerBlock

# token embedding: this is of size (vocab_size, hidden_size)
# and is basically a look up table for the tokens and projecting them into the hidden dim

# positional encoding: this is of size (context_size, hidden_size) because it is of the form
# (idx) -> hidden_dim e.g. it takes the tokens index in the total context size and
# maps it to some hidden dimension vector that represents its position in the sequence


def pe_forward(pe: PositionalEncoding, x: torch.Tensor) -> torch.Tensor:
    """
    Adjust how we run forward method based on type of positional encoding used
    """


@dataclass
class Common:
    hidden_size: int
    context_size: int
    num_layers: int
    vocab_size: int
    weight_tying: bool


@dataclass
class ModelConfig:
    embedding: nn.Embedding
    positional_encoding: PositionalEncoding
    attention: Attention
    ffn: FeedForward
    norm: nn.Module
    transformer_blocks: nn.ModuleList
    head: nn.Linear
    common: Common


class CausalLLM(nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.config = model_config
        self.token_embedding = self.config.embedding
        self.pe = self.config.positional_encoding
        self.attention = self.config.attention
        self.ffn = self.config.ffn
        self.norm = self.config.norm
        self.blocks = self.config.transformer_blocks
        self.head = self.config.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is of size batch, seq_len
        B, S = x.size()

        # get the token embeddings -> project into batch, seq_len, hidden_size
        x = self.token_embedding(x)

        # add positional encoding if relevant
        x = x + self.pe(x) if type(self.pe) == SinusoidalPE else x

        # maybe an optional drop out here

        # iterate through the transformer blocks
        for block in self.blocks:
            x = block(x)

        return self.head(x)

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ):
        """
        Naive generation method taken from https://github.com/karpathy/llama2.c/blob/master/model.py
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.common.context_size
                else idx[:, -self.config.common.context_size :]
            )

            # forward the model to get the logits for the index in the sequence
            # this will produce a tensor of size (batch, seq_len, vocab_size)
            # but we only care about the last one
            logits = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("Inf")
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


if __name__ == "__main__":
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode("It's a good itme to")
    tokens = torch.tensor(tokens).unsqueeze(0)
    print(tokens.size())  # 1 x num_tokens
    torch.manual_seed(42)

    # TODO: read in model parameters from config
    # @dataclass
    # class Config:
    #     hidden_size: int = 512
    #     num_heads_q: int = 8
    #     num_heads_k: int = 8
    #     num_heads_v: int = 8
    #     context_size: int = 512
    #     ffn_size: int = 2048
    #     num_layers: int = 6
    #     vocab_size: int = 50304
    #     dropout: float = 0.1
    #     activation: nn.Module = nn.GELU

    # model_config = load_model_config("gpt2")
    # model = CausalLLM(model_config)
    # model.eval()

    # while tokens.size(1) < 10:
    #     with torch.no_grad():
    #         out = model(tokens)
    #         print(out.size())

    #         logits = out[:, -1, :]
    #         print(logits[:5])

    #         probs = F.softmax(logits, dim=-1)
    #         print(probs.size())
    #         topk_probs, tokp_indices = torch.topk(probs, 50, dim=-1)
    #         selected = torch.multinomial(topk_probs, num_samples=1)
    #         print(selected.size())

    #         next_token = torch.gather(tokp_indices, -1, selected)
    #         tokens = torch.cat([tokens, next_token], dim=1)
    #         print(tokens.size())

    # print(enc.decode(tokens.squeeze().tolist()))
