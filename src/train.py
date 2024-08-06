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
    return OptimizerConfig(lr=3e-4, betas=(0.9, 0.999))


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


def train(model: str, optimizer_config: str, epochs: int):

    # load model config - this should include seq
    model_config: ModelConfig = load_model_config(model)

    # load optimizer parameters
    optimizer_params: OptimizerConfig = load_optimizer_params(optimizer_config)

    # instantiate model
    model = CausalLLM(model_config)

    # instantiate optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=optimizer_params.lr, betas=optimizer_params.betas
    )

    # TODO: we need seq_len, batch_size from model config
    dataset = TokenDataSet("gpt2", 128, "./data/tiny_shakespeare.txt")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # training loop
    for epoch in range(epochs):
        step = 0
        for x, y in data_loader:
            optimizer.zero_grad()
            # shape B, S, V
            output = model(x)
            loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            print(f"step: {step}, Loss: {loss.item()}")
            step += 1

    return model


if __name__ == "__main__":
    train("gpt2", "adamw", 1)
