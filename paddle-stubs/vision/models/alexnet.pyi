from __future__ import annotations

from typing import Any, Optional

import paddle.nn as nn

model_urls: Any

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
    def forward(self, inputs: Any): ...

class AlexNet(nn.Layer):
    num_classes: Any = ...
    def __init__(self, num_classes: int = ...) -> None: ...
    def forward(self, inputs: Any): ...

def alexnet(pretrained: bool = ..., **kwargs: Any): ...
