from enum import Enum
from typing import Any, Literal

import torch.nn as nn
import yaml
from pydantic import BaseModel, Field, field_validator, validator

from model import ModelConfig
from transformer.attention import Attention, Mask
from transformer.positional_encoding import PositionalEncoding, SinusoidalPE
from transformer.transformer import FeedForward, TransformerBlock


class SupportedPE(Enum):
    SINUSOIDAL = "sinusoidal"
    # LEARNABLE = "learnable"
    # ROPE = "rope"


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


TYPE_TO_IMPLEMENTATION = {
    "sinusoidal": SinusoidalPE,
    "gelu": nn.GELU,
    "layer": nn.LayerNorm,
    "batch": nn.BatchNorm1d,
}


class ModelCommon(BaseModel):
    hidden_size: int = Field(512, description="The hidden size of the model")
    context_size: int = Field(1024, description="The context size of the model")
    vocab_size: int = Field(50304, description="The size of the vocabulary")
    num_layers: int = Field(
        6, description="The number of Transformer layers in the model"
    )

    @validator("hidden_size", "context_size", "vocab_size", "num_layers")
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError(f"Must be positive")
        return v


class AttentionConfig(BaseModel):
    hidden_size: int = Field(512, description="The hidden size of the model")
    num_heads_q: int = Field(8, description="Number of heads for query")
    num_heads_k: int = Field(8, description="Number of heads for key")
    num_heads_v: int = Field(8, description="Number of heads for value")
    context_size: int = Field(1024, description="The context size of the model")

    attn_drop: float = Field(0.1, description="Dropout rate for attention")
    output_drop: float = Field(0.1, description="Dropout rate for output")

    mask: SupportedMask = Field(
        SupportedMask.CAUSAL.value, description="The type of mask to use"
    )
    is_causal: bool = Field(True, description="Whether the mask is causal or not")

    @field_validator("num_q_heads", "num_k_heads", "num_v_heads", "hidden_size")
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError(f"Must be positive")
        return v

    @field_validator("attn_drop", "output_drop")
    def must_be_between_0_and_1(cls, v):
        if v < 0 or v > 1:
            raise ValueError(f"Must be between 0 and 1")
        return v


class PEConfig(BaseModel):
    pe_type: SupportedPE = Field(
        SupportedPE.SINUSOIDAL.value,
        description="The type of positional encoding to use",
    )


class FeedForwardConfig(BaseModel):
    ffn_size: int = Field(2048, description="The size of the feed forward network")
    activation_func: SupportedActivations = Field(
        SupportedActivations.GELU.value, description="The activation function to use"
    )
    dropout: float = Field(0.1, description="Dropout rate for feed forward network")

    @validator("hidden_size", "ffn_size")
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError(f"Must be positive")
        return v

    @validator("dropout")
    def must_be_between_0_and_1(cls, v):
        if v < 0 or v > 1:
            raise ValueError(f"Must be between 0 and 1")
        return v


class TransformerBlockConfig(BaseModel):
    norm_placement: SupportedNormPlacements = Field(
        SupportedNormPlacements.PRE.value,
        description="The type of normalization to use",
    )
    norm_type: SupportedNorms = Field(
        SupportedNorms.LAYER.value, description="The type of normalization to use"
    )


def load_config(file_path: str) -> dict[str, Any]:
    with open(file_path, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data


def build_model_config(file_path: str) -> ModelConfig:
    """
    Loads in the model configs and validates values.

    """
    config = load_config(file_path)
    model_common = ModelCommon(**config["common"])
    attention_config = AttentionConfig(**config["attention"])
    pe_config = PEConfig(**config["positional_encoding"])
    ffn_config = FeedForwardConfig(**config["feed_forward"])
    transformer_block_config = TransformerBlockConfig(**config["transformer_block"])

    attn = Attention(
        hidden_size=attention_config.hidden_size,
        num_heads_q=attention_config.num_heads_q,
        num_heads_k=attention_config.num_heads_k,
        num_heads_v=attention_config.num_heads_v,
        context_size=attention_config.context_size,
        mask=Mask(
            attention_config.context_size, attention_config.mask
        ).causal_mask,  # TODO: update this to be dynamic
        attn_drop=attention_config.attn_drop,
        output_drop=attention_config.output_drop,
    )
    pe_model: PositionalEncoding = TYPE_TO_IMPLEMENTATION[pe_config.pe_type]
    pe = pe_model(
        hidden_size=model_common.hidden_size,
        context_size=model_common.context_size,
    )
    activation = TYPE_TO_IMPLEMENTATION[ffn_config.activation_func]
    ffn = FeedForward(
        hidden_size=model_common.hidden_size,
        ffn_size=ffn_config.ffn_size,
        activation=activation,
        output_drop=ffn_config.dropout,
    )
    norm = TYPE_TO_IMPLEMENTATION[transformer_block_config.norm_type]
    block = nn.ModuleList(
        [
            TransformerBlock(
                attention=attn,
                positional_encoding=pe,
                ffn=ffn,
                norm=norm,
                pre_norm=transformer_block_config.norm_placement,
            )
            for _ in range(model_common.num_layers)
        ]
    )
    token_embedding = nn.Embedding(model_common.vocab_size, model_common.hidden_size)
    head = nn.Linear(model_common.hidden_size, model_common.vocab_size)

    return ModelConfig(
        embedding=token_embedding,
        positional_encoding=pe,
        attention=attn,
        ffn=ffn,
        norm=norm,
        transformer_blocks=block,
        head=head,
    )


if __name__ == "__main__":
    config = build_model_config("./configs/olmo.yaml")
    print(config)
