"""
Training loop for a generic LLM model here
"""

import os
from dataclasses import dataclass
from functools import partial
from typing import Any

import click
import tiktoken
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import enable_wrap, size_based_auto_wrap_policy, wrap
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from .model import CausalLLM, ModelConfig
from .utils import (
    SupportedDistStrat,
    TrainingConfig,
    build_model_config,
    build_training_config,
)


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


def setup(rank, world_size, dataset, backend: str = "nccl") -> tuple[int, int]:
    """
    Set up the training state for distributed training if available
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    print(f"Setting up process {rank + 1}/{world_size}")

    # if world_size > 1 then we start a dist process
    if world_size > 1:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        sampler: Sampler = DistributedSampler(
            dataset, rank=rank, num_replicas=world_size
        )
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sampler = None

    return {
        "device": device,
        "sampler": sampler,
        "shuffle": sampler is None,
    }


def distribute_model(model: nn.Module, state: dict, strategy: str) -> nn.Module:
    if state["world_size"] == 1:
        return model.to(state["device"])
    match strategy:
        case SupportedDistStrat.DDP:
            model = DDP(
                model, device_ids=[state["rank"]] if torch.cuda.is_available() else None
            )
        case SupportedDistStrat.FSDP:
            auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=1e5)
            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                cpu_offload=CPUOffload(offload_params=True),
                backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            )
        case SupportedDistStrat.DATA_PARALLEL:
            if torch.cuda.is_available():
                model = DataParallel(model, device_ids=[state["rank"]])
            else:
                model = model
        case _:
            model = model

    return model


def train(
    rank: int,
    world_size: int,
    model_config_path: str,
    training_config_path: str,
    data_path: str,
):
    model_config: ModelConfig = build_model_config(model_config_path)
    training_config: TrainingConfig = build_training_config(training_config_path)
    dataset = TokenDataSet("gpt2", model_config.common.context_size, data_path)

    # set up training state (dist or not)
    state: dict[str, Any] = setup(rank, world_size, dataset)
    state.update({"world_size": world_size, "rank": rank})
    device = state["device"]
    print(f"Rank {rank} using device: {device}")

    # instantiate model, optimizer and data loader
    model = CausalLLM(model_config).to(device)
    optimizer: Optimizer = training_config.partial_optimizer(model.parameters())

    model = distribute_model(model, state, training_config.distributed_strategy)
    worker_batch_size = training_config.batch_size // world_size
    data_loader = DataLoader(
        dataset,
        batch_size=worker_batch_size,
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
            print(f"Rank {rank}, Epoch {epoch}, Step {step}, Loss: {loss.item()}")
            step += 1

    if world_size > 1:
        dist.destroy_process_group()
    return model


@click.command()
@click.option("--model-config-path", type=str, required=True)
@click.option("--training-config-path", type=str, required=True)
@click.option("--data-path", type=str, required=True)
def main(model_config_path: str, training_config_path: str, data_path: str):
    cuda_avaialble = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if cuda_avaialble else 1
    print(f"Found {world_size} {'GPUs' if cuda_avaialble else 'CPU'}")
    mp.spawn(
        train,
        args=(world_size, model_config_path, training_config_path, data_path),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
