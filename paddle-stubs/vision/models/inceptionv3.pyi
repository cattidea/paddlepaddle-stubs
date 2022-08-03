from __future__ import annotations

from typing import Any

from ... import Tensor, nn
from ..ops import ConvNormActivation as ConvNormActivation

class InceptionStem(nn.Layer):
    conv_1a_3x3: Any = ...
    conv_2a_3x3: Any = ...
    conv_2b_3x3: Any = ...
    max_pool: Any = ...
    conv_3b_1x1: Any = ...
    conv_4a_3x3: Any = ...
    def __init__(self) -> None: ...
    def forward(self, x: Any) -> Tensor: ...

class InceptionA(nn.Layer):
    branch1x1: Any = ...
    branch5x5_1: Any = ...
    branch5x5_2: Any = ...
    branch3x3dbl_1: Any = ...
    branch3x3dbl_2: Any = ...
    branch3x3dbl_3: Any = ...
    branch_pool: Any = ...
    branch_pool_conv: Any = ...
    def __init__(self, num_channels: Any, pool_features: Any) -> None: ...
    def forward(self, x: Any) -> Tensor: ...

class InceptionB(nn.Layer):
    branch3x3: Any = ...
    branch3x3dbl_1: Any = ...
    branch3x3dbl_2: Any = ...
    branch3x3dbl_3: Any = ...
    branch_pool: Any = ...
    def __init__(self, num_channels: Any) -> None: ...
    def forward(self, x: Any) -> Tensor: ...

class InceptionC(nn.Layer):
    branch1x1: Any = ...
    branch7x7_1: Any = ...
    branch7x7_2: Any = ...
    branch7x7_3: Any = ...
    branch7x7dbl_1: Any = ...
    branch7x7dbl_2: Any = ...
    branch7x7dbl_3: Any = ...
    branch7x7dbl_4: Any = ...
    branch7x7dbl_5: Any = ...
    branch_pool: Any = ...
    branch_pool_conv: Any = ...
    def __init__(self, num_channels: Any, channels_7x7: Any) -> None: ...
    def forward(self, x: Any) -> Tensor: ...

class InceptionD(nn.Layer):
    branch3x3_1: Any = ...
    branch3x3_2: Any = ...
    branch7x7x3_1: Any = ...
    branch7x7x3_2: Any = ...
    branch7x7x3_3: Any = ...
    branch7x7x3_4: Any = ...
    branch_pool: Any = ...
    def __init__(self, num_channels: Any) -> None: ...
    def forward(self, x: Any) -> Tensor: ...

class InceptionE(nn.Layer):
    branch1x1: Any = ...
    branch3x3_1: Any = ...
    branch3x3_2a: Any = ...
    branch3x3_2b: Any = ...
    branch3x3dbl_1: Any = ...
    branch3x3dbl_2: Any = ...
    branch3x3dbl_3a: Any = ...
    branch3x3dbl_3b: Any = ...
    branch_pool: Any = ...
    branch_pool_conv: Any = ...
    def __init__(self, num_channels: Any) -> None: ...
    def forward(self, x: Any) -> Tensor: ...

class InceptionV3(nn.Layer):
    num_classes: Any = ...
    with_pool: Any = ...
    layers_config: Any = ...
    inception_stem: Any = ...
    inception_block_list: Any = ...
    avg_pool: Any = ...
    dropout: Any = ...
    fc: Any = ...
    def __init__(
        self,
        num_classes: int = ...,
        with_pool: bool = ...,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    __call__ = forward

def inception_v3(
    pretrained: bool = ...,
    *,
    num_classes: int = ...,
    with_pool: bool = ...,
) -> InceptionV3: ...
