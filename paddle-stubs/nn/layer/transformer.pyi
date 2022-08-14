from __future__ import annotations

from typing import Any, NamedTuple, Optional, Sequence

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
        kdim: Optional[int] = ...,
        vdim: Optional[int] = ...,
        need_weights: bool = ...,
        weight_attr: Optional[ParamAttr] = ...,
        bias_attr: Optional[ParamAttr] = ...,
    ) -> None: ...
    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = ...,
        value: Optional[Tensor] = ...,
        attn_mask: Optional[Tensor] = ...,
        cache: Optional[MultiHeadAttention.Cache | MultiHeadAttention.StaticCache] = ...,
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
        attn_dropout: Optional[float] = ...,
        act_dropout: Optional[float] = ...,
        normalize_before: bool = ...,
        weight_attr: Optional[ParamAttr | Sequence[ParamAttr]] = ...,
        bias_attr: Optional[ParamAttr | Sequence[ParamAttr] | bool] = ...,
    ) -> None: ...
    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = ...,
        cache: Optional[Tensor] = ...,
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
        norm: Optional[LayerNorm] = ...,
    ) -> None: ...
    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = ...,
        cache: Optional[list[MultiHeadAttention.Cache | MultiHeadAttention.StaticCache]] = ...,
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
        attn_dropout: Optional[float] = ...,
        act_dropout: Optional[float] = ...,
        normalize_before: bool = ...,
        weight_attr: Optional[ParamAttr | Sequence[ParamAttr]] = ...,
        bias_attr: Optional[ParamAttr | Sequence[ParamAttr] | bool] = ...,
    ) -> None: ...
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = ...,
        memory_mask: Optional[Tensor] = ...,
        cache: Optional[tuple[MultiHeadAttention.Cache, ...] | tuple[MultiHeadAttention.StaticCache, ...]] = ...,
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
        norm: Optional[LayerNorm] = ...,
    ) -> None: ...
    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = ...,
        memory_mask: Optional[Tensor] = ...,
        cache: Optional[list[MultiHeadAttention.Cache] | list[MultiHeadAttention.StaticCache]] = ...,
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
        attn_dropout: Optional[float] = ...,
        act_dropout: Optional[float] = ...,
        normalize_before: bool = ...,
        weight_attr: Optional[ParamAttr | Sequence[ParamAttr]] = ...,
        bias_attr: Optional[ParamAttr | Sequence[ParamAttr] | bool] = ...,
        custom_encoder: Optional[Layer] = ...,
        custom_decoder: Optional[Layer] = ...,
    ) -> None: ...
    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = ...,
        tgt_mask: Optional[Tensor] = ...,
        memory_mask: Optional[Tensor] = ...,
    ) -> Tensor: ...
    __call__ = forward
