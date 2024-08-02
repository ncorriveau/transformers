from typing import Literal

from pydantic import BaseModel, Field, validator


class AttentionConfig(BaseModel):
    hidden_size: int = Field(512, description="The hidden size of the model")
    num_heads_q: int = Field(8, description="Number of heads for query")
    num_heads_k: int = Field(8, description="Number of heads for key")
    num_heads_v: int = Field(8, description="Number of heads for value")
    context_size: int = Field(1024, description="The context size of the model")

    attn_drop: float = Field(0.1, description="Dropout rate for attention")
    output_drop: float = Field(0.1, description="Dropout rate for output")

    mask: Literal["sliding_window", "global"] = Field(
        "causal", description="The type of mask to use"
    )
    is_causal: bool = Field(True, description="Whether the mask is causal or not")

    @validator("num_q_heads", "num_k_heads", "num_v_heads", "hidden_size")
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError(f"Must be positive")
        return v

    @validator("attn_drop", "output_drop")
    def must_be_between_0_and_1(cls, v):
        if v < 0 or v > 1:
            raise ValueError(f"Must be between 0 and 1")
        return v


class PEConfig(BaseModel):
    hidden_size: int = Field(512, description="The hidden size of the model")
    context_size: int = Field(1024, description="The context size of the model")
    pe_type: Literal["sinusoidal", "learnable", "rope"] = Field(
        "sinusoidal", description="The type of positional encoding to use"
    )

    @validator("hidden_size", "context_size")
    def must_be_positive(cls, v):
        if v <= 0:
            raise ValueError(f"Must be positive")
        return v


class FeedForwardConfig(BaseModel):
    hidden_size: int = Field(512, description="The hidden size of the model")
    ffn_size: int = Field(2048, description="The size of the feed forward network")
    activation_func: Literal["relu", "gelu", "swiglu"] = Field(
        "gelu", description="The activation function to use"
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
    norm_placement: Literal["pre", "post"] = Field(
        "pre", description="The type of normalization to use"
    )
    norm_type: Literal["layer", "batch"] = Field(
        "layer", description="The type of normalization to use"
    )
