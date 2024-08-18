"""
Training loop for a generic LLM model here
"""

import gc
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
    ShardingStrategy
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

def fsdp_wrap_policy(module, recurse, nonwrapped_numel):
    if recurse:
        return True
    return nonwrapped_numel >= 1e5 and not isinstance(module, FSDP)

def get_module_params_numel(module):
    return sum(p.numel() for p in module.parameters())

def wrap_module(module, wrap_policy, fsdp_config):
    if wrap_policy(module, False, get_module_params_numel(module)):
        return FSDP(module, **fsdp_config)
    for name, child in module.named_children():
        setattr(module, name, wrap_module(child, wrap_policy, fsdp_config))
    return module


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

# def fsdp_wrap_policy(module: nn.Module, recurse: bool, nonwrapped_numel: float) -> bool:
#     if recurse:
#         return True  # always recurse
#     return nonwrapped_numel >= 1e5  # wrap if numel >= 100M

def distribute_model(model: nn.Module, state: dict, strategy: str) -> nn.Module:
    if state["world_size"] == 1:
        return model.to(state["device"])
    match strategy:
        case SupportedDistStrat.DDP:
            model = DDP(
                model, device_ids=[state["rank"]] if torch.cuda.is_available() else None
            )
        case SupportedDistStrat.FSDP:
            fsdp_config = {
            "sharding_strategy": ShardingStrategy.SHARD_GRAD_OP,
            "cpu_offload": None,  # Disable CPU offloading for now
            "backward_prefetch": BackwardPrefetch.BACKWARD_PRE,
            "device_id": state["rank"] if torch.cuda.is_available() else None,
            "use_orig_params": True,
            }
            wrap_policy = partial(fsdp_wrap_policy)
            
            # Wrap the token embedding separately
            model.token_embedding = FSDP(model.token_embedding, **fsdp_config)
            
            # Wrap each transformer block
            for i, block in enumerate(model.blocks):
                model.blocks[i] = wrap_module(block, wrap_policy, fsdp_config)
            
            # Wrap the entire model
            model = FSDP(model, **fsdp_config)
        
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
    model.train()
    worker_batch_size = training_config.batch_size // world_size
    data_loader = DataLoader(
        dataset,
        batch_size=worker_batch_size,
        sampler=state["sampler"],
        shuffle=state["shuffle"],
    )

    # training loop
    for epoch in range(training_config.epochs):
        print(f"Rank {rank} CUDA Memory Stats:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"  Cached:    {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        ddp_loss = torch.zeros(2).to(device)
        if state["sampler"]:
            state["sampler"].set_epoch(epoch)
        step = 0
        for x, y in data_loader:
            optimizer.zero_grad()
            # shape B, S, Vocab size
            x, y = x.to(device), y.to(device)
            output: torch.Tensor = model(x)
            loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1))
            
            # Synchronize processes for cleaner output
            if step % 100 == 0 and rank == 0:
                print(f"Rank {rank}, Epoch {epoch}, Step {step}, Loss: {loss.item()}")
            
            loss.backward()
            optimizer.step()
            ddp_loss[0] += loss.item()
            ddp_loss[1] += len(x)
            step += 1
        
        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        torch.cuda.empty_cache()
        gc.collect()
        
        if rank == 0:
            print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, ddp_loss[0] / ddp_loss[1]))

            # with torch.no_grad():
            #     new_output = model(x)
            #     new_loss = F.cross_entropy(new_output.view(-1, new_output.size(-1)), y.view(-1))
            #     print(f"Rank {rank}, Epoch {epoch}, Step {step}, Loss after step: {new_loss.item()}")
            
            

    if world_size > 1:
        dist.destroy_process_group()
    return model


@click.command()
@click.option("--model-config-path", type=str, required=True)
@click.option("--training-config-path", type=str, required=True)
@click.option("--data-path", type=str, required=True)
def main(model_config_path: str, training_config_path: str, data_path: str):
    cuda_avaialble = torch.cuda.is_available()
    if cuda_avaialble:
        torch.cuda.empty_cache()

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
