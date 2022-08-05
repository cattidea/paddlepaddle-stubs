from __future__ import annotations

from typing import Any, Optional

import numpy as np
from paddle.framework import ParamAttr
from paddle.incubate.nn import functional as incubate_f
from paddle.nn import Layer
from paddle.nn import functional as F

from ...._typing import Tensor

class FusedBiasDropoutResidualLayerNorm(Layer):
    def __init__(
        self,
        embed_dim: int,
        dropout_rate: float = ...,
        weight_attr: Optional[ParamAttr] = ...,
        bias_attr: Optional[ParamAttr] = ...,
        epsilon: float = 1e-5,
        name: Optional[str] = ...,
    ) -> None: ...
    def forward(self, x, residual) -> Tensor: ...
    def extra_repr(self) -> str: ...

class FusedMultiHeadAttention(Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout_rate=0.5,
        attn_dropout_rate=0.5,
        kdim=None,
        vdim=None,
        normalize_before=False,
        need_weights=False,
        qkv_weight_attr=None,
        qkv_bias_attr=None,
        linear_weight_attr=None,
        linear_bias_attr=None,
        pre_ln_scale_attr=None,
        pre_ln_bias_attr=None,
        ln_scale_attr=None,
        ln_bias_attr=None,
        epsilon=1e-5,
        nranks=1,
        ring_id=-1,
        name=None,
    ): ...
    def forward(self, query, key=None, value=None, attn_mask=None, cache=None): ...
    def extra_repr(self) -> str: ...

class FusedFeedForward(Layer):
    def __init__(
        self,
        d_model,
        dim_feedforward,
        dropout_rate=0.1,
        epsilon=1e-05,
        activation="relu",
        act_dropout_rate=None,
        normalize_before=False,
        linear1_weight_attr=None,
        linear1_bias_attr=None,
        linear2_weight_attr=None,
        linear2_bias_attr=None,
        ln1_scale_attr=None,
        ln1_bias_attr=None,
        ln2_scale_attr=None,
        ln2_bias_attr=None,
        nranks=1,
        ring_id=-1,
        name=None,
    ): ...
    def forward(self, src, cache=None) -> Tensor: ...
    def extra_repr(self) -> str:
        name_str = ", name={}".format(self.name) if self.name else ""
        return "d_model={}, dim_feedforward={}, dropout_rate={}, epsilon={}, activation={}, act_dropout_rate={}, normalize_before={}, dtype={}{}".format(
            self._d_model,
            self._dim_feedforward,
            self._dropout_rate,
            self._epsilon,
            self._act_method,
            self._act_dropout_rate,
            self._normalize_before,
            self._dtype,
            name_str,
        )
    def _amp_decorate(self, dtype): ...

class FusedTransformerEncoderLayer(Layer):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward,
        dropout_rate=0.1,
        activation="relu",
        attn_dropout_rate=None,
        act_dropout_rate=None,
        normalize_before=False,
        weight_attr=None,
        bias_attr=None,
    ): ...
    def forward(self, src, src_mask=None, cache=None): ...

class FusedTransformer(Layer):
    def __init__(
        self,
        d_model: int = ...,
        nhead: int = ...,
        num_encoder_layers: int = ...,
        num_decoder_layers: int = ...,
        dim_feedforward: int = ...,
        dropout: float = ...,
        activation: str = ...,
        attn_dropout: Optional[Any] = ...,
        act_dropout: Optional[Any] = ...,
        normalize_before: bool = ...,
        weight_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        custom_encoder: Optional[Any] = ...,
        custom_decoder: Optional[Any] = ...,
    ): ...
    def forward(
        self, src, tgt, src_mask: Optional[Any] = ..., tgt_mask: Optional[Any] = ..., memory_mask: Optional[Any] = ...
    ): ...

class FusedMultiTransformer(Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dim_feedforward,
        dropout_rate=0.0,
        activation="gelu",
        normalize_before=True,
        ln_scale_attrs: Optional[ParamAttr] = None,
        ln_bias_attrs: Optional[ParamAttr] = None,
        qkv_weight_attrs: Optional[ParamAttr] = None,
        qkv_bias_attrs: Optional[ParamAttr] = None,
        linear_weight_attrs: Optional[ParamAttr] = None,
        linear_bias_attrs: Optional[ParamAttr] = None,
        ffn_ln_scale_attrs: Optional[ParamAttr] = None,
        ffn_ln_bias_attrs: Optional[ParamAttr] = None,
        ffn1_weight_attrs: Optional[ParamAttr] = None,
        ffn1_bias_attrs: Optional[ParamAttr] = None,
        ffn2_weight_attrs: Optional[ParamAttr] = None,
        ffn2_bias_attrs: Optional[ParamAttr] = None,
        epsilon: float = ...,
        num_layers: int = ...,
        nranks: int = ...,
        trans_qkvw: bool = ...,
        ring_id: int = ...,
        name: Optional[str] = ...,
    ) -> None: ...
    def forward(self, src, attn_mask=None, caches=None, time_step=None): ...
