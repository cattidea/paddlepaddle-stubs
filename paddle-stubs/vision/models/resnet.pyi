from __future__ import annotations

from typing import Any, Optional

from ... import Tensor, nn

class BasicBlock(nn.Layer):
    expansion: int = ...
    conv1: Any = ...
    bn1: Any = ...
    relu: Any = ...
    conv2: Any = ...
    bn2: Any = ...
    downsample: Any = ...
    stride: Any = ...
    def __init__(
        self,
        inplanes: Any,
        planes: Any,
        stride: int = ...,
        downsample: Optional[Any] = ...,
        groups: int = ...,
        base_width: int = ...,
        dilation: int = ...,
        norm_layer: Optional[Any] = ...,
    ) -> None: ...
    def forward(self, x: Any) -> Tensor: ...

class BottleneckBlock(nn.Layer):
    expansion: int = ...
    conv1: Any = ...
    bn1: Any = ...
    conv2: Any = ...
    bn2: Any = ...
    conv3: Any = ...
    bn3: Any = ...
    relu: Any = ...
    downsample: Any = ...
    stride: Any = ...
    def __init__(
        self,
        inplanes: Any,
        planes: Any,
        stride: int = ...,
        downsample: Optional[Any] = ...,
        groups: int = ...,
        base_width: int = ...,
        dilation: int = ...,
        norm_layer: Optional[Any] = ...,
    ) -> None: ...
    def forward(self, x: Any) -> Tensor: ...

class ResNet(nn.Layer):
    def __init__(
        self,
        block: type[BasicBlock] | type[BottleneckBlock],
        depth: int = ...,
        width: int = ...,
        num_classes: int = ...,
        with_pool: bool = ...,
        groups: int = ...,
    ) -> None: ...
    def forward(self, x: Any) -> Tensor: ...

def resnet18(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
    groups: int = ...,
) -> ResNet: ...
def resnet34(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
    groups: int = ...,
) -> ResNet: ...
def resnet50(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
    groups: int = ...,
) -> ResNet: ...
def resnet101(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
    groups: int = ...,
) -> ResNet: ...
def resnet152(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
    groups: int = ...,
) -> ResNet: ...
def resnext50_32x4d(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
    groups: int = ...,
) -> ResNet: ...
def resnext50_64x4d(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
    groups: int = ...,
) -> ResNet: ...
def resnext101_32x4d(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
    groups: int = ...,
) -> ResNet: ...
def resnext101_64x4d(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
    groups: int = ...,
) -> ResNet: ...
def resnext152_32x4d(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
    groups: int = ...,
) -> ResNet: ...
def resnext152_64x4d(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
    groups: int = ...,
) -> ResNet: ...
def wide_resnet50_2(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
    groups: int = ...,
) -> ResNet: ...
def wide_resnet101_2(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
    groups: int = ...,
) -> ResNet: ...
