"""
Training loop for a generic LLM model here
"""

from dataclasses import dataclass

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import CausalLLM, ModelConfig, load_model_config


@dataclass
class OptimizerConfig:
    lr: float
    betas: float


# TODO fill out to read in from config yaml
def load_optimizer_params(optimizer_config: str):
    return OptimizerConfig(lr=1e-3, betas=(0.9, 0.999))


def create_data_set(
    tokenizer: str,
    batch_size: int = 32,
    seq_len: int = 128,
    data_path: str = "./data/tiny_shakespeare.txt",
):
    """
    taken from karpathy's preferred way of loading / training on small
    data set
    """
    with open(data_path, "r") as f:
        data = f.read()

    enc = tiktoken.get_encoding(tokenizer)
    tokens = enc.encode(data)
    buf = torch.tensor(tokens[: batch_size * seq_len + 1])

    x = buf[:-1].view(batch_size, seq_len)
    y = buf[1:].view(batch_size, seq_len)
    return x, y


def train(model: str, optimizer_config: str, epochs: int):
    # load model
    model_config: ModelConfig = load_model_config(model)

    # load optimizer parameters
    optimizer_params: OptimizerConfig = load_optimizer_params(optimizer_config)

    # instantiate model
    model = CausalLLM(model_config)

    # instantiate optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=optimizer_params.lr, betas=optimizer_params.betas
    )

    X, Y = create_data_set("gpt2")
    dataset = torch.utils.data.TensorDataset(X, Y)

    # training loop
    for _ in range(epochs):
        optimizer.zero_grad()

        # size, B, S, Vocab Size
        output = model(X)

        loss = F.cross_entropy(output.view(-1, output.size(-1)), Y.view(-1))
        print(loss)
        loss.backward()
        # optimizer.step()
        # print(f"Epoch: {epoch}, Loss: {loss.item()}")

    return model


if __name__ == "__main__":
    train("gpt2", "adamw", 1)
