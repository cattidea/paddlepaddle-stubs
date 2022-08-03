from __future__ import annotations

from typing import Any, Optional

from ... import Tensor, nn

class ConvPoolLayer(nn.Layer):
    relu: Any = ...
    def __init__(
        self,
        input_channels: Any,
        output_channels: Any,
        filter_size: Any,
        stride: Any,
        padding: Any,
        stdv: Any,
        groups: int = ...,
        act: Optional[Any] = ...,
    ) -> None: ...
    def forward(self, inputs: Any) -> Tensor: ...

class AlexNet(nn.Layer):
    def __init__(self, num_classes: int = ...) -> None: ...
    def forward(self, inputs: Tensor) -> Tensor: ...
    __call__ = forward

def alexnet(pretrained: bool = ..., *, num_classes: int = ...) -> AlexNet: ...
