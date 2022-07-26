from __future__ import annotations

from typing import Any

import paddle.nn as nn

model_urls: Any

def xavier(channels: Any, filter_size: Any): ...

class ConvLayer(nn.Layer):
    def __init__(
        self, num_channels: Any, num_filters: Any, filter_size: Any, stride: int = ..., groups: int = ...
    ) -> None: ...
    def forward(self, inputs: Any): ...

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
    def forward(self, inputs: Any): ...

class GoogLeNet(nn.Layer):
    num_classes: Any = ...
    with_pool: Any = ...
    def __init__(self, num_classes: int = ..., with_pool: bool = ...) -> None: ...
    def forward(self, inputs: Any): ...

def googlenet(pretrained: bool = ..., **kwargs: Any): ...
