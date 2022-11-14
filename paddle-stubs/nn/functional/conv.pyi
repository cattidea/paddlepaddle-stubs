from __future__ import annotations

from typing import Any, Optional

from ...device import get_cudnn_version as get_cudnn_version
from ...fluid import dygraph_utils as dygraph_utils
from ...fluid.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ...fluid.layer_helper import LayerHelper as LayerHelper
from ...fluid.layers import nn as nn
from ...fluid.layers.utils import convert_to_list as convert_to_list
from ...framework import ParamAttr as ParamAttr
from ...static import Variable as Variable
from ...tensor.manipulation import squeeze as squeeze
from ...tensor.manipulation import unsqueeze as unsqueeze
from ...tensor.math import add as add

def conv1d(
    x: Any,
    weight: Any,
    bias: Any | None = ...,
    stride: int = ...,
    padding: int = ...,
    dilation: int = ...,
    groups: int = ...,
    data_format: str = ...,
    name: str | None = ...,
): ...
def conv2d(
    x: Any,
    weight: Any,
    bias: Any | None = ...,
    stride: int = ...,
    padding: int = ...,
    dilation: int = ...,
    groups: int = ...,
    data_format: str = ...,
    name: str | None = ...,
): ...
def conv1d_transpose(
    x: Any,
    weight: Any,
    bias: Any | None = ...,
    stride: int = ...,
    padding: int = ...,
    output_padding: int = ...,
    groups: int = ...,
    dilation: int = ...,
    output_size: Any | None = ...,
    data_format: str = ...,
    name: str | None = ...,
): ...
def conv2d_transpose(
    x: Any,
    weight: Any,
    bias: Any | None = ...,
    stride: int = ...,
    padding: int = ...,
    output_padding: int = ...,
    dilation: int = ...,
    groups: int = ...,
    output_size: Any | None = ...,
    data_format: str = ...,
    name: str | None = ...,
): ...
def conv3d(
    x: Any,
    weight: Any,
    bias: Any | None = ...,
    stride: int = ...,
    padding: int = ...,
    dilation: int = ...,
    groups: int = ...,
    data_format: str = ...,
    name: str | None = ...,
): ...
def conv3d_transpose(
    x: Any,
    weight: Any,
    bias: Any | None = ...,
    stride: int = ...,
    padding: int = ...,
    output_padding: int = ...,
    groups: int = ...,
    dilation: int = ...,
    output_size: Any | None = ...,
    data_format: str = ...,
    name: str | None = ...,
): ...
