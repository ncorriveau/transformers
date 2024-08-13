"""
Training loop for a generic LLM model here
"""

import os
from dataclasses import dataclass

import click
import tiktoken
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
)
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

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


def setup(dataset, backend: str = "nccl") -> tuple[int, int]:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    # if world_size > 1 then we start a dist process
    if world_size - 1 and rank:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        sampler: Sampler = DistributedSampler(
            dataset, rank=rank, num_replicas=world_size
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sampler = None
    return {
        "rank": rank,
        "world_size": world_size,
        "device": device,
        "sampler": sampler,
        "shuffle": sampler is None,
    }


def distribute_model(model: nn.Module, state: dict, strategy: str) -> nn.Module:
    if state["world_size"] == 1:
        return model
    match strategy:
        case "ddp":
            model = DDP(model, device_ids=[state["rank"]])
        case "fsdp":
            model = FSDP(
                model,
                cpu_offload=CPUOffload(offload_params=True),
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            )
        case "data_parallel":
            model = DataParallel(model, device_ids=[state["rank"]])
        case _:
            model = model

    return model


@click.command()
@click.option("--model-config-path", type=str, required=True)
@click.option("--training-config-path", type=str, required=True)
@click.option("--data-path", type=str, required=True)
def train(model_config_path: str, training_config_path: str, data_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config: ModelConfig = build_model_config(model_config_path)
    training_config: TrainingConfig = build_training_config(training_config_path)
    dataset = TokenDataSet("gpt2", model_config.common.context_size, data_path)
    # process state
    state = setup()

    # instantiate model, optimizer and data loader
    model = CausalLLM(model_config).to(state["device"])
    optimizer: Optimizer = training_config.partial_optimizer(model.parameters())

    model = distribute_model(model, state, training_config.distributed_strategy)

    data_loader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        sampler=state["sampler"],
        shuffle=state["shuffle"],
    )

    # training loop
    for epoch in range(training_config.epochs):
        if state["sampler"]:
            state["sampler"].set_epoch(epoch)
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
