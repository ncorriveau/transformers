"""
Training loop for a generic LLM model here
"""

from dataclasses import dataclass

import click
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .model import CausalLLM, ModelConfig
from .utils import TrainingConfig, build_model_config, build_training_config


class TokenDataSet(torch.utils.data.Dataset):
    def __init__(self, tokenizer: str, seq_len: int, data_path: str):
        try:
            with open(data_path, "r") as f:
                data = f.read()

        except Exception as e:
            print(f"Error loading data: {e}")
            raise e

        enc = tiktoken.get_encoding(tokenizer)
        self.tokens = enc.encode(data)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x), torch.tensor(y)

    def __repr__(self):
        return f"TokenDataSet with {len(self.tokens)} tokens"


@click.command()
@click.option("--model-config-path", type=str, required=True)
@click.option("--training-config-path", type=str, required=True)
@click.option("--data-path", type=str, required=True)
def train(model_config_path: str, training_config_path: str, data_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config: ModelConfig = build_model_config(model_config_path)
    training_config: TrainingConfig = build_training_config(training_config_path)

    # instantiate model, optimizer and data loader
    model = CausalLLM(model_config).to(device)
    optimizer: Optimizer = training_config.partial_optimizer(model.parameters())
    dataset = TokenDataSet("gpt2", model_config.common.context_size, data_path)
    data_loader = DataLoader(
        dataset, batch_size=training_config.batch_size, shuffle=True
    )

    # training loop
    for _ in range(training_config.epochs):
        step = 0
        for x, y in data_loader:
            optimizer.zero_grad()
            # shape B, S, Vocab size
            x, y = x.to(device), y.to(device)
            output: torch.Tensor = model(x)
            loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            print(f"step: {step}, Loss: {loss.item()}")
            step += 1

    return model


if __name__ == "__main__":
    train()
