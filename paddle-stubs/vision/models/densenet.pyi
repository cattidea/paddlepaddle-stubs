from __future__ import annotations

from typing import Any

from ... import Tensor, nn

class BNACConvLayer(nn.Layer):
    def __init__(
        self,
        num_channels: Any,
        num_filters: Any,
        filter_size: Any,
        stride: int = ...,
        pad: int = ...,
        groups: int = ...,
        act: str = ...,
    ) -> None: ...
    def forward(self, input: Any) -> Tensor: ...

class DenseLayer(nn.Layer):
    dropout: Any = ...
    bn_ac_func1: Any = ...
    bn_ac_func2: Any = ...
    dropout_func: Any = ...
    def __init__(self, num_channels: Any, growth_rate: Any, bn_size: Any, dropout: Any) -> None: ...
    def forward(self, input: Any) -> Tensor: ...

class DenseBlock(nn.Layer):
    dropout: Any = ...
    dense_layer_func: Any = ...
    def __init__(
        self,
        num_channels: Any,
        num_layers: Any,
        bn_size: Any,
        growth_rate: Any,
        dropout: Any,
        name: str | None = ...,
    ) -> None: ...
    def forward(self, input: Any) -> Tensor: ...

class TransitionLayer(nn.Layer):
    conv_ac_func: Any = ...
    pool2d_avg: Any = ...
    def __init__(self, num_channels: Any, num_output_features: Any) -> None: ...
    def forward(self, input: Any) -> Tensor: ...

class ConvBNLayer(nn.Layer):
    def __init__(
        self,
        num_channels: Any,
        num_filters: Any,
        filter_size: Any,
        stride: int = ...,
        pad: int = ...,
        groups: int = ...,
        act: str = ...,
    ) -> None: ...
    def forward(self, input: Any) -> Tensor: ...

class DenseNet(nn.Layer):
    def __init__(
        self,
        layers: int = ...,
        bn_size: int = ...,
        dropout: float = ...,
        num_classes: int = ...,
        with_pool: bool = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    __call__ = forward

def densenet121(
    pretrained: bool = ...,
    *,
    layers: int = ...,
    bn_size: int = ...,
    dropout: float = ...,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> DenseNet: ...
def densenet161(
    pretrained: bool = ...,
    *,
    layers: int = ...,
    bn_size: int = ...,
    dropout: float = ...,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> DenseNet: ...
def densenet169(
    pretrained: bool = ...,
    *,
    layers: int = ...,
    bn_size: int = ...,
    dropout: float = ...,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> DenseNet: ...
def densenet201(
    pretrained: bool = ...,
    *,
    layers: int = ...,
    bn_size: int = ...,
    dropout: float = ...,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> DenseNet: ...
def densenet264(
    pretrained: bool = ...,
    *,
    layers: int = ...,
    bn_size: int = ...,
    dropout: float = ...,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> DenseNet: ...
