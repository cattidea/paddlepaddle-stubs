from __future__ import annotations

from typing import Any, Optional

from typing_extensions import Literal

from ... import Tensor, nn

def create_activation_layer(act: Any) -> Optional[type[nn.ReLU] | type[nn.Swish]]: ...
def channel_shuffle(x: Tensor, groups: int) -> Tensor: ...

class InvertedResidual(nn.Layer):
    def __init__(self, in_channels: Any, out_channels: Any, stride: Any, activation_layer: Any = ...) -> None: ...
    def forward(self, inputs: Any) -> Tensor: ...

class InvertedResidualDS(nn.Layer):
    def __init__(self, in_channels: Any, out_channels: Any, stride: Any, activation_layer: Any = ...) -> None: ...
    def forward(self, inputs: Any) -> Tensor: ...

class ShuffleNetV2(nn.Layer):
    def __init__(
        self,
        scale: float = ...,
        act: Optional[Literal["relu", "swish"]] = ...,
        num_classes: int = ...,
        with_pool: bool = ...,
    ) -> None: ...
    def forward(self, inputs: Any) -> Tensor: ...

def shufflenet_v2_x0_25(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> ShuffleNetV2: ...
def shufflenet_v2_x0_33(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> ShuffleNetV2: ...
def shufflenet_v2_x0_5(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> ShuffleNetV2: ...
def shufflenet_v2_x1_0(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> ShuffleNetV2: ...
def shufflenet_v2_x1_5(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> ShuffleNetV2: ...
def shufflenet_v2_x2_0(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> ShuffleNetV2: ...
def shufflenet_v2_swish(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> ShuffleNetV2: ...
