from __future__ import annotations

from typing import Any

from ... import Tensor, nn

class MakeFireConv(nn.Layer):
    def __init__(self, input_channels: Any, output_channels: Any, filter_size: Any, padding: int = ...) -> None: ...
    def forward(self, x: Any) -> Tensor: ...

class MakeFire(nn.Layer):
    def __init__(
        self, input_channels: Any, squeeze_channels: Any, expand1x1_channels: Any, expand3x3_channels: Any
    ) -> None: ...
    def forward(self, inputs: Any) -> Tensor: ...

class SqueezeNet(nn.Layer):
    def __init__(
        self,
        version: Any,
        num_classes: int = ...,
        with_pool: bool = ...,
    ) -> None: ...
    def forward(self, inputs: Any) -> Tensor: ...

def squeezenet1_0(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> SqueezeNet: ...
def squeezenet1_1(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> SqueezeNet: ...
