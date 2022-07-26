from __future__ import annotations

from typing import Any

import paddle.nn as nn

from . import utils as utils

class Identity(nn.Layer):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def forward(self, input: Any): ...

def fuse_conv_bn(model: Any) -> None: ...
def fuse_layers(model: Any, layers_to_fuse: Any, inplace: bool = ...): ...

types_to_fusion_method: Any
