from __future__ import annotations

from typing import Any

from ... import Tensor, nn

class VGG(nn.Layer):
    def __init__(
        self,
        features: nn.Layer,
        num_classes: int = ...,
        with_pool: bool = ...,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    __call__ = forward

def make_layers(cfg: Any, batch_norm: bool = ...) -> nn.Layer: ...

cfgs: Any

def vgg11(
    pretrained: bool = ...,
    batch_norm: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> VGG: ...
def vgg13(
    pretrained: bool = ...,
    batch_norm: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> VGG: ...
def vgg16(
    pretrained: bool = ...,
    batch_norm: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> VGG: ...
def vgg19(
    pretrained: bool = ...,
    batch_norm: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> VGG: ...
