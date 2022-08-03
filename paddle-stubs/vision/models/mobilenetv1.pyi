from __future__ import annotations

from typing import Any

from ... import Tensor, nn

class DepthwiseSeparable(nn.Layer):
    def __init__(
        self, in_channels: Any, out_channels1: Any, out_channels2: Any, num_groups: Any, stride: Any, scale: Any
    ) -> None: ...
    def forward(self, x: Any) -> Tensor: ...

class MobileNetV1(nn.Layer):
    def __init__(
        self,
        scale: float = ...,
        num_classes: int = ...,
        with_pool: bool = ...,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    __call__ = forward

def mobilenet_v1(
    pretrained: bool = ...,
    scale: float = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> MobileNetV1: ...
