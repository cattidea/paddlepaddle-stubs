from __future__ import annotations

from typing import Any

from ... import Tensor, nn
from ..ops import ConvNormActivation as ConvNormActivation

model_urls: Any

class InvertedResidual(nn.Layer):
    stride: Any = ...
    use_res_connect: Any = ...
    conv: Any = ...
    def __init__(self, inp: Any, oup: Any, stride: Any, expand_ratio: Any, norm_layer: Any = ...) -> None: ...
    def forward(self, x: Any) -> Tensor: ...

class MobileNetV2(nn.Layer):
    def __init__(
        self,
        scale: float = ...,
        num_classes: int = ...,
        with_pool: bool = ...,
    ) -> None: ...
    def forward(self, x: Any) -> Tensor: ...

def mobilenet_v2(
    pretrained: bool = ...,
    scale: float = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> MobileNetV2: ...
