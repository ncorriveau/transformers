from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, List, Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import yaml
from packaging import version
from pydantic import BaseModel, Field, field_validator, model_validator
from torch.cuda.amp import GradScaler
from typing_extensions import Self

from .model import Common, ModelConfig
from .transformer.attention import Attention, Mask
from .transformer.norms import RMSNorm
from .transformer.positional_encoding import (
    PositionalEncoding,
    RotaryEmbedding,
    SinusoidalPE,
)
from .transformer.transformer import FeedForward, SwiGLU, TransformerBlock


class SupportedPE(Enum):
    SINUSOIDAL = "sinusoidal"
    ROPE = "rope"
    # LEARNABLE = "learnable"


class SupportedActivations(Enum):
    RELU = "relu"
    GELU = "gelu"
    SWIGLU = "swiglu"


class SupportedNorms(Enum):
    LAYER = "layer"
    BATCH = "batch"


class SupportedNormPlacements(Enum):
    PRE = "pre"
    POST = "post"


class SupportedMask(Enum):
    CAUSAL = "causal"
    SLIDING_WINDOW = "sliding_window"
    GLOBAL = "global"


class SupportedDistStrat(Enum):
    DDP = "ddp"
    FSDP = "fsdp"
    DATA_PARALLEL = "data_parallel"


TYPE_TO_IMPLEMENTATION = {
    "sinusoidal": SinusoidalPE,
    "rope": RotaryEmbedding,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swiglu": SwiGLU,
    "layer": nn.LayerNorm,
    "batch": nn.BatchNorm1d,
    "rms": RMSNorm,
}


class ModelCommon(BaseModel):
    hidden_size: int = Field(..., gt=0, description="The hidden size of the model")
    context_size: int = Field(..., gt=0, description="The context size of the model")
    vocab_size: int = Field(..., gt=0, description="The size of the vocabulary")
    num_layers: int = Field(
        ..., gt=0, description="The number of Transformer layers in the model"
    )
    weight_tying: bool = Field(
        False, description="Whether to tie the weights of the token embedding and head"
    )


class AttentionConfig(BaseModel):
    num_heads_q: int = Field(..., gt=0, description="Number of query heads")
    num_heads_k: int = Field(..., gt=0, description="Number of key heads")
    num_heads_v: int = Field(..., gt=0, description="Number of value heads")

    attn_drop: float = Field(..., description="Dropout rate for attention")
    output_drop: float = Field(..., description="Dropout rate for output")

    mask: SupportedMask = Field(
        SupportedMask.CAUSAL.value, description="The type of mask to use"
    )
    is_causal: bool = Field(..., description="Whether the mask is causal or not")

    @field_validator("attn_drop", "output_drop")
    @classmethod
    def must_be_between_0_and_1(cls, v):
        if v < 0 or v > 1:
            raise ValueError(f"Must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def check_head_compatability(self) -> Self:
        assert (
            self.num_heads_q % self.num_heads_k == 0
        ), "Number of query heads must be divisible by the number of key heads"
        assert (
            self.num_heads_q % self.num_heads_v == 0
        ), "Number of query heads must be divisible by the number of value heads"


class PEConfig(BaseModel):
    pe_type: SupportedPE = Field(
        ...,
        description="The type of positional encoding to use",
    )


class FeedForwardConfig(BaseModel):
    ffn_size: int = Field(..., gt=0, description="The size of the feed forward network")
    activation_func: SupportedActivations = Field(
        ..., description="The activation function to use"
    )
    dropout: float = Field(..., description="Dropout rate for feed forward network")

    @field_validator("dropout")
    @classmethod
    def must_be_between_0_and_1(cls, v: int):
        if v < 0 or v > 1:
            raise ValueError(f"Must be between 0 and 1")
        return v


class NormConfig(BaseModel):
    norm_type: SupportedNorms = Field(
        ..., description="The type of normalization to use"
    )


class TransformerBlockConfig(BaseModel):
    transformer_block: List[str] = Field(
        ..., description="The component order of the transformer block"
    )


@dataclass
class TrainingConfig:
    partial_optimizer: Callable
    optimizer_name: str
    optimizer_args: Dict[str, Any]
    partial_scheduler: Callable
    scheduler_name: str
    scheduler_args: Dict[str, Any]
    clip_grad_norm: float
    grad_scaler: GradScaler
    batch_size: int
    epochs: int
    dtype: torch.dtype
    distributed_strategy: str
    use_mp: bool
    compile: bool


class TrainConfig(BaseModel):
    optimizer_name: str = Field(
        ..., description="The name of the optimizer to use e.g. AdamW"
    )
    optimizer_args: Dict[str, Union[float, List[float], str]] = Field(
        default_factory=dict
    )
    scheduler_name: str = Field(description="The name of the lr scheduler to use")
    scheduler_args: Dict[str, Union[float, List[float], str]] = Field(
        default_factory=dict
    )
    clip_grad_norm: float = Field(
        1.0, description="The value to clip the gradient norm to, defaults to 0 = off"
    )
    use_grad_scaler: bool = Field(
        True, description="Whether to use a gradient scaler - cuda must be available"
    )
    batch_size: int = Field(..., gt=0, description="The batch size to use")
    epochs: int = Field(..., gt=0, description="The number of epochs to train for")
    distributed_strategy: SupportedDistStrat = Field(
        None, description="The distributed strategy to use"
    )
    dtype: str = Field("float32", description="The dtype to use for training")
    use_mp: bool = Field(True, description="Whether to use mixed precision training")
    compile: bool = Field(
        True, description="Whether to compile the model - cuda must be available"
    )

    @field_validator("optimizer_name")
    @classmethod
    def validate_optimizer_name(cls, v: str):
        if not hasattr(optim, v):
            raise ValueError(f"'{v}' is not a valid optimizer in torch.optim")
        return v

    @field_validator("scheduler_name")
    @classmethod
    def validate_scheduler_name(cls, v: str):
        if not hasattr(lr_scheduler, v):
            raise ValueError(
                f"'{v}' is not a learning rate scheduler in torch.optim.lr_scheduler"
            )
        return v

    @field_validator("optimizer_args")
    @classmethod
    def parse_numeric_args(cls, v: Dict[str, Any]):
        for key, value in v.items():
            if isinstance(value, str):
                try:
                    v[key] = float(value)
                except ValueError:
                    pass  # Keep as string if it can't be converted to float
        return v

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str):
        if not hasattr(torch, v):
            raise ValueError(f"'{v}' is not a valid dtype in torch")
        return getattr(torch, v)

    @field_validator("compile")
    @classmethod
    def validate_compile(cls, v: bool):
        if version.parse(torch.__version__) < version.parse("2.0.0") and v:
            print(f"Warning: 'compile' is only available in torch >= 2.0.0")
            return False
        return v

    class Config:
        extra = "allow"  # This allows for additional fields in the args


def load_config(file_path: str) -> dict[str, Any]:
    with open(file_path, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


def build_pe_args(
    pe_type: SupportedPE,
    model_common: ModelCommon,
    attention_config: AttentionConfig,
    device: torch.device,
) -> dict:
    """helper function to build the args for the positional encoding. Probably
    over kill for now but gives flexibility for new types"""
    extra_args = dict()
    if pe_type == SupportedPE.ROPE:
        extra_args.update(
            {"num_q_k_heads": attention_config.num_heads_q, "device": device}
        )
    return {
        "hidden_size": model_common.hidden_size,
        "context_size": model_common.context_size,
        **extra_args,
    }


# yes this is ugly, maybe a better way but we'll see if it works for now.
def build_model_config(file_path: str, device: torch.device) -> ModelConfig:
    """
    Loads in the model configs and validates values.

    """
    config = load_config(file_path)
    model_common = ModelCommon(**config["common"])
    attention_config = AttentionConfig(**config["attention"])
    pe_config = PEConfig(**config["positional_encoding"])
    ffn_config = FeedForwardConfig(**config["feed_forward"])
    norm_config = NormConfig(**config["norm"])
    block_config = TransformerBlockConfig(transformer_block=config["transformer_block"])

    # make sure the components in the block are defined in the config
    valid_components = any(
        [
            component in config.keys()
            for component in block_config.transformer_block
            if component != "skip"
        ]
    )
    assert valid_components, "Transformer components must be defined in the config file"

    # TODO: we need to make the mask obj dynamic here.
    pe_model: PositionalEncoding = TYPE_TO_IMPLEMENTATION[pe_config.pe_type.value]
    pe_args = build_pe_args(pe_config.pe_type, model_common, attention_config, device)
    pe = pe_model(**pe_args)
    rotation = None if pe_config.pe_type != SupportedPE.ROPE else pe

    attn = Attention(
        hidden_size=model_common.hidden_size,
        num_heads_q=attention_config.num_heads_q,
        num_heads_k=attention_config.num_heads_k,
        num_heads_v=attention_config.num_heads_v,
        context_size=model_common.context_size,
        mask=Mask(model_common.context_size).causal_mask,
        rotation=rotation,
        attn_drop=attention_config.attn_drop,
        output_drop=attention_config.output_drop,
    )
    activation = TYPE_TO_IMPLEMENTATION[ffn_config.activation_func.value]
    if ffn_config.activation_func == SupportedActivations.SWIGLU:
        activation = activation(hidden_size=model_common.hidden_size)

    ffn = FeedForward(
        hidden_size=model_common.hidden_size,
        ffn_size=ffn_config.ffn_size,
        activation=activation,
        output_drop=ffn_config.dropout,
    )
    norm = TYPE_TO_IMPLEMENTATION[norm_config.norm_type.value](model_common.hidden_size)
    transformer_block = block_config.transformer_block
    block = nn.ModuleList(
        [
            TransformerBlock(
                attention=attn,
                positional_encoding=pe,
                ffn=ffn,
                norm=norm,
                transformer_config=transformer_block,
            )
            for _ in range(model_common.num_layers)
        ]
    )
    token_embedding = nn.Embedding(model_common.vocab_size, model_common.hidden_size)
    head = nn.Linear(model_common.hidden_size, model_common.vocab_size)
    if model_common.weight_tying:
        head.weight = token_embedding.weight
    common = Common(**model_common.model_dump())

    return ModelConfig(
        embedding=token_embedding,
        positional_encoding=pe,
        attention=attn,
        ffn=ffn,
        norm=norm,
        transformer_blocks=block,
        head=head,
        common=common,
    )


def build_training_config(training_config: str) -> TrainingConfig:
    config = load_config(training_config)
    dtype_is_f16 = config.get("dtype") == "float16"
    cuda_available = torch.cuda.is_available()

    validated_config = TrainConfig(**config)
    optimizer = getattr(optim, validated_config.optimizer_name)
    partial_optimizer = partial(optimizer, **validated_config.optimizer_args)

    scheduler = getattr(lr_scheduler, validated_config.scheduler_name)
    scheduler_optimizer = partial(scheduler, **validated_config.scheduler_args)

    return TrainingConfig(
        partial_optimizer=partial_optimizer,
        optimizer_name=validated_config.optimizer_name,
        optimizer_args=validated_config.optimizer_args,
        partial_scheduler=scheduler_optimizer,
        scheduler_name=validated_config.scheduler_name,
        scheduler_args=validated_config.scheduler_args,
        clip_grad_norm=validated_config.clip_grad_norm,
        grad_scaler=GradScaler(
            enabled=validated_config.use_grad_scaler and dtype_is_f16 and cuda_available
        ),
        batch_size=validated_config.batch_size,
        epochs=validated_config.epochs,
        dtype=validated_config.dtype,
        distributed_strategy=validated_config.distributed_strategy,
        use_mp=validated_config.use_mp,
        compile=validated_config.compile and cuda_available,
    )


if __name__ == "__main__":
    model_config = build_model_config("./configs/models/gpt2.yaml")
    # training_config = build_training_config("./configs/models/gpt2.yaml")
    print(model_config)
    # file_path = "./configs/models/gpt2.yaml"
    # config = load_config(file_path)
    # print(config)
