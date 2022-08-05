from __future__ import annotations

from typing import Optional

from paddle import ParamAttr
from paddle.incubate.nn import functional as F
from paddle.nn import Layer

class FusedLinear(Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_attr: ParamAttr = ...,
        bias_attr: ParamAttr = ...,
        transpose_weight: bool = ...,
        name: Optional[str] = ...,
    ): ...
    def forward(self, input): ...
    __call__ = forward
