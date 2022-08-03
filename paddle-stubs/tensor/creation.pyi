from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .. import Tensor
from .._typing import DTypeLike, NestedNumbericSequence, Numberic, ShapeLike
from ..fluid.data_feeder import check_dtype as check_dtype
from ..fluid.data_feeder import check_type as check_type
from ..fluid.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ..fluid.data_feeder import convert_dtype as convert_dtype
from ..fluid.framework import in_dygraph_mode as in_dygraph_mode
from ..fluid.layer_helper import LayerHelper as LayerHelper
from ..fluid.layers import linspace as linspace
from ..fluid.layers import tensor as tensor
from ..fluid.layers import utils as utils
from ..framework import OpProtoHolder as OpProtoHolder
from ..framework import convert_np_dtype_to_dtype_ as convert_np_dtype_to_dtype_
from ..framework import core as core
from ..framework import dygraph_only as dygraph_only
from ..static import Variable as Variable
from ..static import device_guard as device_guard

def to_tensor(
    data: Numberic | NestedNumbericSequence | np.ndarray[Any, Any] | Tensor,
    dtype: Optional[str | np.dtype[Any]] = ...,  # TODO: paddle.dtype
    place: Optional[Any] = ...,  # TODO: CPUPlace | CUDAPinnedPlace | CUDAPlace
    stop_gradient: bool = ...,
) -> Tensor: ...
def full_like(x: Any, fill_value: Any, dtype: Optional[Any] = ..., name: Optional[Any] = ...): ...
def ones(
    shape: ShapeLike,
    dtype: Optional[DTypeLike] = ...,
    name: Optional[str] = ...,
) -> Tensor: ...
def ones_like(x: Any, dtype: Optional[Any] = ..., name: Optional[Any] = ...): ...
def zeros(shape: Any, dtype: Optional[Any] = ..., name: Optional[Any] = ...): ...
def zeros_like(x: Any, dtype: Optional[Any] = ..., name: Optional[Any] = ...): ...
def eye(num_rows: Any, num_columns: Optional[Any] = ..., dtype: Optional[Any] = ..., name: Optional[Any] = ...): ...
def full(
    shape: ShapeLike,
    fill_value: bool | int | float | Tensor,
    dtype: Optional[Any] = ...,
    name: Optional[Any] = ...,
) -> Tensor: ...
def arange(
    start: int = ..., end: Optional[Any] = ..., step: int = ..., dtype: Optional[Any] = ..., name: Optional[Any] = ...
): ...
def tril(x: Any, diagonal: int = ..., name: Optional[Any] = ...): ...
def triu(x: Any, diagonal: int = ..., name: Optional[Any] = ...): ...
def meshgrid(*args: Any, **kwargs: Any): ...
def diagflat(x: Any, offset: int = ..., name: Optional[Any] = ...): ...
def diag(x: Any, offset: int = ..., padding_value: int = ..., name: Optional[Any] = ...): ...
def empty(shape: Any, dtype: Optional[Any] = ..., name: Optional[Any] = ...): ...
def empty_like(x: Any, dtype: Optional[Any] = ..., name: Optional[Any] = ...): ...
def assign(x: Any, output: Optional[Any] = ...): ...
def clone(x: Any, name: Optional[Any] = ...): ...
def complex(real: Any, imag: Any, name: Optional[Any] = ...): ...
