from __future__ import annotations

from typing import Any, Optional

from ...base.data_feeder import check_type as check_type
from ...base.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ...base.layers import LayerHelper as LayerHelper
from ...base.layers import utils as utils
from ...tensor.manipulation import squeeze as squeeze
from ...tensor.manipulation import unsqueeze as unsqueeze

def avg_pool1d(
    x: Any,
    kernel_size: Any,
    stride: Any | None = ...,
    padding: int = ...,
    exclusive: bool = ...,
    ceil_mode: bool = ...,
    name: str | None = ...,
): ...
def avg_pool2d(
    x: Any,
    kernel_size: Any,
    stride: Any | None = ...,
    padding: int = ...,
    ceil_mode: bool = ...,
    exclusive: bool = ...,
    divisor_override: Any | None = ...,
    data_format: str = ...,
    name: str | None = ...,
): ...
def avg_pool3d(
    x: Any,
    kernel_size: Any,
    stride: Any | None = ...,
    padding: int = ...,
    ceil_mode: bool = ...,
    exclusive: bool = ...,
    divisor_override: Any | None = ...,
    data_format: str = ...,
    name: str | None = ...,
): ...
def max_pool1d(
    x: Any,
    kernel_size: Any,
    stride: Any | None = ...,
    padding: int = ...,
    return_mask: bool = ...,
    ceil_mode: bool = ...,
    name: str | None = ...,
): ...
def max_unpool1d(
    x: Any,
    indices: Any,
    kernel_size: Any,
    stride: Any | None = ...,
    padding: int = ...,
    data_format: str = ...,
    output_size: Any | None = ...,
    name: str | None = ...,
): ...
def max_unpool2d(
    x: Any,
    indices: Any,
    kernel_size: Any,
    stride: Any | None = ...,
    padding: int = ...,
    data_format: str = ...,
    output_size: Any | None = ...,
    name: str | None = ...,
): ...
def max_unpool3d(
    x: Any,
    indices: Any,
    kernel_size: Any,
    stride: Any | None = ...,
    padding: int = ...,
    data_format: str = ...,
    output_size: Any | None = ...,
    name: str | None = ...,
): ...
def max_pool2d(
    x: Any,
    kernel_size: Any,
    stride: Any | None = ...,
    padding: int = ...,
    return_mask: bool = ...,
    ceil_mode: bool = ...,
    data_format: str = ...,
    name: str | None = ...,
): ...
def max_pool3d(
    x: Any,
    kernel_size: Any,
    stride: Any | None = ...,
    padding: int = ...,
    return_mask: bool = ...,
    ceil_mode: bool = ...,
    data_format: str = ...,
    name: str | None = ...,
): ...
def adaptive_avg_pool1d(x: Any, output_size: Any, name: str | None = ...): ...
def adaptive_avg_pool2d(x: Any, output_size: Any, data_format: str = ..., name: str | None = ...): ...
def adaptive_avg_pool3d(x: Any, output_size: Any, data_format: str = ..., name: str | None = ...): ...
def adaptive_max_pool1d(x: Any, output_size: Any, return_mask: bool = ..., name: str | None = ...): ...
def adaptive_max_pool2d(x: Any, output_size: Any, return_mask: bool = ..., name: str | None = ...): ...
def adaptive_max_pool3d(x: Any, output_size: Any, return_mask: bool = ..., name: str | None = ...): ...
