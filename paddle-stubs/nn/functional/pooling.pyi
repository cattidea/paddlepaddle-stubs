from __future__ import annotations

from typing import Any, Optional

from ...fluid.data_feeder import check_type as check_type
from ...fluid.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ...fluid.layers import LayerHelper as LayerHelper
from ...fluid.layers import utils as utils
from ...tensor.manipulation import squeeze as squeeze
from ...tensor.manipulation import unsqueeze as unsqueeze

def avg_pool1d(
    x: Any,
    kernel_size: Any,
    stride: Optional[Any] = ...,
    padding: int = ...,
    exclusive: bool = ...,
    ceil_mode: bool = ...,
    name: Optional[str] = ...,
): ...
def avg_pool2d(
    x: Any,
    kernel_size: Any,
    stride: Optional[Any] = ...,
    padding: int = ...,
    ceil_mode: bool = ...,
    exclusive: bool = ...,
    divisor_override: Optional[Any] = ...,
    data_format: str = ...,
    name: Optional[str] = ...,
): ...
def avg_pool3d(
    x: Any,
    kernel_size: Any,
    stride: Optional[Any] = ...,
    padding: int = ...,
    ceil_mode: bool = ...,
    exclusive: bool = ...,
    divisor_override: Optional[Any] = ...,
    data_format: str = ...,
    name: Optional[str] = ...,
): ...
def max_pool1d(
    x: Any,
    kernel_size: Any,
    stride: Optional[Any] = ...,
    padding: int = ...,
    return_mask: bool = ...,
    ceil_mode: bool = ...,
    name: Optional[str] = ...,
): ...
def max_unpool1d(
    x: Any,
    indices: Any,
    kernel_size: Any,
    stride: Optional[Any] = ...,
    padding: int = ...,
    data_format: str = ...,
    output_size: Optional[Any] = ...,
    name: Optional[str] = ...,
): ...
def max_unpool2d(
    x: Any,
    indices: Any,
    kernel_size: Any,
    stride: Optional[Any] = ...,
    padding: int = ...,
    data_format: str = ...,
    output_size: Optional[Any] = ...,
    name: Optional[str] = ...,
): ...
def max_unpool3d(
    x: Any,
    indices: Any,
    kernel_size: Any,
    stride: Optional[Any] = ...,
    padding: int = ...,
    data_format: str = ...,
    output_size: Optional[Any] = ...,
    name: Optional[str] = ...,
): ...
def max_pool2d(
    x: Any,
    kernel_size: Any,
    stride: Optional[Any] = ...,
    padding: int = ...,
    return_mask: bool = ...,
    ceil_mode: bool = ...,
    data_format: str = ...,
    name: Optional[str] = ...,
): ...
def max_pool3d(
    x: Any,
    kernel_size: Any,
    stride: Optional[Any] = ...,
    padding: int = ...,
    return_mask: bool = ...,
    ceil_mode: bool = ...,
    data_format: str = ...,
    name: Optional[str] = ...,
): ...
def adaptive_avg_pool1d(x: Any, output_size: Any, name: Optional[str] = ...): ...
def adaptive_avg_pool2d(x: Any, output_size: Any, data_format: str = ..., name: Optional[str] = ...): ...
def adaptive_avg_pool3d(x: Any, output_size: Any, data_format: str = ..., name: Optional[str] = ...): ...
def adaptive_max_pool1d(x: Any, output_size: Any, return_mask: bool = ..., name: Optional[str] = ...): ...
def adaptive_max_pool2d(x: Any, output_size: Any, return_mask: bool = ..., name: Optional[str] = ...): ...
def adaptive_max_pool3d(x: Any, output_size: Any, return_mask: bool = ..., name: Optional[str] = ...): ...
