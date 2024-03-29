from __future__ import annotations

from functools import partial as partial
from functools import reduce as reduce
from typing import Any, Optional

from paddle import framework as framework
from paddle.base.data_feeder import convert_dtype as convert_dtype
from paddle.base.layers.utils import flatten as flatten
from paddle.base.layers.utils import map_structure as map_structure
from paddle.base.layers.utils import pack_sequence_as as pack_sequence_as
from paddle.device import get_cudnn_version as get_cudnn_version
from paddle.device import get_device as get_device
from paddle.nn import Layer
from paddle.nn import LayerList as LayerList

def resnet_unit(
    x: Any,
    filter_x: Any,
    scale_x: Any,
    bias_x: Any,
    mean_x: Any,
    var_x: Any,
    z: Any,
    filter_z: Any,
    scale_z: Any,
    bias_z: Any,
    mean_z: Any,
    var_z: Any,
    stride: Any,
    stride_z: Any,
    padding: Any,
    dilation: Any,
    groups: Any,
    momentum: Any,
    eps: Any,
    data_format: Any,
    fuse_add: Any,
    has_shortcut: Any,
    use_global_stats: Any,
    is_test: Any,
    act: Any,
): ...

class ResNetUnit(Layer):
    filter_x: Any = ...
    scale_x: Any = ...
    bias_x: Any = ...
    mean_x: Any = ...
    var_x: Any = ...
    filter_z: Any = ...
    scale_z: Any = ...
    bias_z: Any = ...
    mean_z: Any = ...
    var_z: Any = ...
    def __init__(
        self,
        num_channels_x: Any,
        num_filters: Any,
        filter_size: Any,
        stride: int = ...,
        momentum: float = ...,
        eps: float = ...,
        data_format: str = ...,
        act: str = ...,
        fuse_add: bool = ...,
        has_shortcut: bool = ...,
        use_global_stats: bool = ...,
        is_test: bool = ...,
        filter_x_attr: Any | None = ...,
        scale_x_attr: Any | None = ...,
        bias_x_attr: Any | None = ...,
        moving_mean_x_name: str | None = ...,
        moving_var_x_name: str | None = ...,
        num_channels_z: int = ...,
        stride_z: int = ...,
        filter_z_attr: Any | None = ...,
        scale_z_attr: Any | None = ...,
        bias_z_attr: Any | None = ...,
        moving_mean_z_name: str | None = ...,
        moving_var_z_name: str | None = ...,
    ): ...
    def forward(self, x: Any, z: Any | None = ...): ...
