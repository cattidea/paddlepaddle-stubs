from __future__ import annotations

from typing import Any

from ... import Tensor, nn
from ...framework import ParamAttr

def xavier(channels: int, filter_size: int) -> ParamAttr: ...

class ConvLayer(nn.Layer):
    def __init__(
        self, num_channels: Any, num_filters: Any, filter_size: Any, stride: int = ..., groups: int = ...
    ) -> None: ...
    def forward(self, inputs: Any) -> Tensor: ...

class Inception(nn.Layer):
    def __init__(
        self,
        input_channels: Any,
        output_channels: Any,
        filter1: Any,
        filter3R: Any,
        filter3: Any,
        filter5R: Any,
        filter5: Any,
        proj: Any,
    ) -> None: ...
    def forward(self, inputs: Any) -> Tensor: ...

class GoogLeNet(nn.Layer):
    def __init__(
        self,
        num_classes: int = ...,
        with_pool: bool = ...,
    ) -> None: ...
    def forward(self, inputs: Tensor) -> tuple[Tensor, Tensor, Tensor]: ...
    __call__ = forward

def googlenet(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> GoogLeNet: ...
