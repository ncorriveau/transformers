"""
Training loop for a generic LLM model here
"""

import os
from contextlib import nullcontext
from functools import partial
from itertools import islice
from typing import Any

import click
import numpy as np
import tiktoken
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from .model import CausalLLM, ModelConfig
from .utils import (
    SupportedDistStrat,
    TrainingConfig,
    build_model_config,
    build_training_config,
)


def setup_checkpoint_dir() -> str:
    # get project root dir
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoints_dir = os.path.join(root_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    return checkpoints_dir


class TokenDataSet(torch.utils.data.Dataset):
    def __init__(self, tokenizer: str, seq_len: int, data_path: str):
        try:
            with open(data_path, 'r') as f:
                data = f.read()

        except Exception as e:
            print(f'Error loading data: {e}')
            raise e

        self.enc = tiktoken.get_encoding(tokenizer)
        self.tokens = self.enc.encode(data)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x), torch.tensor(y)

    def __repr__(self):
        return f'TokenDataSet with {len(self.tokens)} tokens'

    def encode_sentence(self, sample_sentence: str) -> torch.Tensor:
        tokens = self.enc.encode(sample_sentence)
        return torch.tensor(tokens).unsqueeze(0)

    @staticmethod
    def get_train_test_split(dataset: torch.utils.data.Dataset, ratio: float = 0.8):
        train, test = torch.utils.data.random_split(dataset, [ratio, 1 - ratio])
        return train, test


def setup(rank, world_size, backend: str = 'nccl') -> tuple[int, int]:
    """
    Set up the training state for distributed training if available
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    print(f'Setting up process {rank + 1}/{world_size}')

    # if world_size > 1 then we start a dist process
    if world_size > 1:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        device = torch.device(f'cuda:{rank}')
        device_type = 'cuda'
        torch.cuda.set_device(device)
    else:
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_type)

    return {
        'device': device,
        'device_type': device_type,
    }


def distribute_model(model: nn.Module, state: dict, strategy: str) -> nn.Module:
    if state['world_size'] == 1:
        return model.to(state['device'])
    match strategy:
        case SupportedDistStrat.DDP:
            model = DDP(
                model, device_ids=[state['rank']] if torch.cuda.is_available() else None
            )
        # TODO: get this to work
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
                model = DataParallel(model, device_ids=[state['rank']])
            else:
                model = model
        case _:
            model = model

    return model


@torch.no_grad()
def get_val_loss(
    model: CausalLLM, val_data: torch.Tensor, context, device: torch.device
):
    model.eval()
    val_loss = val_size = 0
    val_data_iter = iter(val_data)
    eval_slice = islice(val_data_iter, 10)
    for x, y in eval_slice:
        x, y = x.to(device), y.to(device)
        with context:
            output = model(x)
        loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1))

        # loss is already avg across batch
        val_loss += loss.item() * x.size(0)
        val_size += x.size(0)

    model.train()
    return val_loss / val_size if val_size > 0 else 0


def patch_numpy():
    if not hasattr(np, 'int'):
        np.int = int
    if not hasattr(np, 'float'):
        np.float = float
    if not hasattr(np, 'bool'):
        np.bool = bool
    if not hasattr(np, 'object'):
        np.object = object
    if not hasattr(np, 'str'):
        np.str = str


def train(
    rank: int,
    world_size: int,
    model_config_path: str,
    training_config_path: str,
    data_path: str,
    checkpoint_path: str = None,
):
    # initial set up of training state
    state: dict[str, Any] = setup(rank, world_size)
    model_config: ModelConfig = build_model_config(model_config_path, state['device'])
    training_config: TrainingConfig = build_training_config(training_config_path)
    dataset = TokenDataSet('gpt2', model_config.common.context_size, data_path)
    check_dir = setup_checkpoint_dir()
    sampler = None

    if world_size > 1:
        sampler: Sampler = DistributedSampler(
            dataset, rank=rank, num_replicas=world_size
        )

    state.update(
        {
            'world_size': world_size,
            'rank': rank,
            'sampler': sampler,
            'shuffle': sampler is None,
        }
    )
    device = state['device']
    print(f'Rank {rank} using device: {device}')

    # from karpathy's nanogpt
    ctx = (
        nullcontext()
        if state['device_type'] == 'cpu' or not training_config.use_mp
        else torch.autocast(
            device_type=state['device_type'], dtype=training_config.dtype
        )
    )

    # instantiate model, optimizer and data loader
    model = CausalLLM(model_config).to(device)
    optimizer: Optimizer = training_config.partial_optimizer(model.parameters())
    scheduler: LRScheduler = training_config.partial_scheduler(optimizer)
    scaler: GradScaler = training_config.grad_scaler

    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Checkpoint file {checkpoint_path} not found')

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_loss = checkpoint['best_val_loss']
        print(f'Loaded checkpoint from {checkpoint_path}')

    if training_config.compile:
        # this is a hack to deal with old versions of numpy and newer torch w compile
        patch_numpy()
        print('Compiling model... this may take a minute')
        model = torch.compile(model)
        print('Model compiled')

    model: CausalLLM = distribute_model(
        model, state, training_config.distributed_strategy
    )
    worker_batch_size = training_config.batch_size // world_size

    # TODO: investigate memory pinning
    train, test = dataset.get_train_test_split(dataset)
    train_loader = DataLoader(
        train,
        batch_size=worker_batch_size,
        sampler=state['sampler'],
        shuffle=state['shuffle'],
    )
    test_loader = DataLoader(
        test,
        batch_size=worker_batch_size,
        sampler=state['sampler'],
        shuffle=state['shuffle'],
    )

    test_phrase = dataset.encode_sentence('It is a good time to').to(device)
    best_loss = np.inf
    for epoch in range(training_config.epochs):
        if state['sampler']:
            state['sampler'].set_epoch(epoch)
        step = 0
        for x, y in train_loader:
            optimizer.zero_grad()

            # shape B, S, V
            x, y = x.to(device), y.to(device)

            # forward pass in mixed precision
            with ctx:
                output: torch.Tensor = model(x)
                loss = F.cross_entropy(output.view(-1, output.size(-1)), y.view(-1))

            # these will just call optimizer.step() and loss.backward()
            # if enable is False in the grad scaler
            scaler.scale(loss).backward()
            if training_config.clip_grad_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), training_config.clip_grad_norm
                )

            scaler.step(optimizer)
            scaler.update()

            print(f'Rank {rank}, Epoch {epoch}, Step {step}, Loss: {loss.item()}')
            # run sample output on our test sentence
            if step % 10 == 0 and step > 0 and rank == 0:
                # TODO: implement get_val_loss
                val_loss = get_val_loss(model, test_loader, ctx, device)
                print(f'Validation loss: {val_loss}')
                if val_loss < best_loss and step > 0:
                    best_loss = val_loss
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_config': model_config_path,
                        'training_config': training_config_path,
                        'iter_num': step,
                        'epoch': epoch,
                        'best_val_loss': best_loss,
                    }
                    torch.save(checkpoint, os.path.join(check_dir, 'best_model.pth'))
                    print(f'Saved best model at epoch {epoch} step {step}')

                with torch.no_grad():
                    model.eval()
                    generated = model.generate(test_phrase, 10, top_k=50)
                    print(dataset.enc.decode(generated.squeeze().cpu().numpy()))
                    model.train()

            step += 1

        scheduler.step()

    if world_size > 1:
        dist.destroy_process_group()
    return model


@click.command()
@click.option('--model-config-path', type=str, required=True)
@click.option('--training-config-path', type=str, required=True)
@click.option('--data-path', type=str, required=True)
@click.option(
    '--checkpoint-path',
    default=None,
    type=str,
    required=False,
    help='Path to previously saved checkpoint',
)
def main(
    model_config_path: str,
    training_config_path: str,
    data_path: str,
    checkpoint_path: str = None,
):
    cuda_avaialble = torch.cuda.is_available()
    world_size = torch.cuda.device_count() if cuda_avaialble else 1
    print(f"Found {world_size} {'GPUs' if cuda_avaialble else 'CPU'}")
    mp.spawn(
        train,
        args=(
            world_size,
            model_config_path,
            training_config_path,
            data_path,
            checkpoint_path,
        ),
        nprocs=world_size,
        join=True,
    )


if __name__ == '__main__':
    main()
