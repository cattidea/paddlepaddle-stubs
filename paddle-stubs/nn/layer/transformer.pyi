from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

from ... import tensor as tensor
from ..._typing import Tensor
from ...framework import ParamAttr
from .. import Layer as Layer
from .. import LayerList as LayerList
from .common import Dropout as Dropout
from .common import Linear as Linear
from .norm import LayerNorm

class MultiHeadAttention(Layer):
    class Cache(NamedTuple):
        k: Tensor
        v: Tensor

    class StaticCache(NamedTuple):
        k: Tensor
        v: Tensor
    embed_dim: Any = ...
    kdim: Any = ...
    vdim: Any = ...
    num_heads: Any = ...
    dropout: Any = ...
    need_weights: Any = ...
    head_dim: Any = ...
    q_proj: Any = ...
    k_proj: Any = ...
    v_proj: Any = ...
    out_proj: Any = ...
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = ...,
        kdim: int | None = ...,
        vdim: int | None = ...,
        need_weights: bool = ...,
        weight_attr: ParamAttr | None = ...,
        bias_attr: ParamAttr | None = ...,
    ) -> None: ...
    def forward(
        self,
        query: Tensor,
        key: Tensor | None = ...,
        value: Tensor | None = ...,
        attn_mask: Tensor | None = ...,
        cache: MultiHeadAttention.Cache | MultiHeadAttention.StaticCache | None = ...,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]: ...
    __call__ = forward

class TransformerEncoderLayer(Layer):
    normalize_before: Any = ...
    self_attn: Any = ...
    linear1: Any = ...
    dropout: Any = ...
    linear2: Any = ...
    norm1: Any = ...
    norm2: Any = ...
    dropout1: Any = ...
    dropout2: Any = ...
    activation: Any = ...
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = ...,
        activation: str = ...,
        attn_dropout: float | None = ...,
        act_dropout: float | None = ...,
        normalize_before: bool = ...,
        weight_attr: ParamAttr | Sequence[ParamAttr] | None = ...,
        bias_attr: ParamAttr | Sequence[ParamAttr] | bool | None = ...,
    ) -> None: ...
    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = ...,
        cache: Tensor | None = ...,
    ) -> Tensor | tuple[Tensor, Tensor]: ...
    __call__ = forward

class TransformerEncoder(Layer):
    layers: Any = ...
    num_layers: Any = ...
    norm: Any = ...
    def __init__(
        self,
        encoder_layer: Layer,
        num_layers: int,
        norm: LayerNorm | None = ...,
    ) -> None: ...
    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = ...,
        cache: list[MultiHeadAttention.Cache | MultiHeadAttention.StaticCache] | None = ...,
    ) -> Tensor | tuple[Tensor, Tensor]: ...
    __call__ = forward

class TransformerDecoderLayer(Layer):
    normalize_before: Any = ...
    self_attn: Any = ...
    cross_attn: Any = ...
    linear1: Any = ...
    dropout: Any = ...
    linear2: Any = ...
    norm1: Any = ...
    norm2: Any = ...
    norm3: Any = ...
    dropout1: Any = ...
    dropout2: Any = ...
    dropout3: Any = ...
    activation: Any = ...
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = ...,
        activation: str = ...,
        attn_dropout: float | None = ...,
        act_dropout: float | None = ...,
        normalize_before: bool = ...,
        weight_attr: ParamAttr | Sequence[ParamAttr] | None = ...,
        bias_attr: ParamAttr | Sequence[ParamAttr] | bool | None = ...,
    ) -> None: ...
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = ...,
        memory_mask: Tensor | None = ...,
        cache: tuple[MultiHeadAttention.Cache, ...] | tuple[MultiHeadAttention.StaticCache, ...] | None = ...,
    ) -> Tensor | tuple[Tensor, tuple[MultiHeadAttention.Cache, MultiHeadAttention.StaticCache]]: ...
    __call__ = forward

class TransformerDecoder(Layer):
    layers: Any = ...
    num_layers: Any = ...
    norm: Any = ...
    def __init__(
        self,
        decoder_layer: Layer,
        num_layers: int,
        norm: LayerNorm | None = ...,
    ) -> None: ...
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = ...,
        memory_mask: Tensor | None = ...,
        cache: list[MultiHeadAttention.Cache] | list[MultiHeadAttention.StaticCache] | None = ...,
    ) -> Tensor | tuple[Tensor, Tensor]: ...
    __call__ = forward

class Transformer(Layer):
    encoder: Any = ...
    decoder: Any = ...
    d_model: Any = ...
    nhead: Any = ...
    def __init__(
        self,
        d_model: int = ...,
        nhead: int = ...,
        num_encoder_layers: int = ...,
        num_decoder_layers: int = ...,
        dim_feedforward: int = ...,
        dropout: float = ...,
        activation: str = ...,
        attn_dropout: float | None = ...,
        act_dropout: float | None = ...,
        normalize_before: bool = ...,
        weight_attr: ParamAttr | Sequence[ParamAttr] | None = ...,
        bias_attr: ParamAttr | Sequence[ParamAttr] | bool | None = ...,
        custom_encoder: Layer | None = ...,
        custom_decoder: Layer | None = ...,
    ) -> None: ...
    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Tensor | None = ...,
        tgt_mask: Tensor | None = ...,
        memory_mask: Tensor | None = ...,
    ) -> Tensor: ...
    __call__ = forward
