from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

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
    data: Numberic | NestedNumbericSequence | npt.NDArray[Any] | Tensor,
    dtype: str | np.dtype[Any] | None = ...,  # TODO: paddle.dtype
    place: Any | None = ...,  # TODO: CPUPlace | CUDAPinnedPlace | CUDAPlace
    stop_gradient: bool = ...,
) -> Tensor: ...
def full_like(x: Any, fill_value: Any, dtype: Any | None = ..., name: str | None = ...): ...
def ones(
    shape: ShapeLike,
    dtype: DTypeLike | None = ...,
    name: str | None = ...,
) -> Tensor: ...
def ones_like(x: Any, dtype: Any | None = ..., name: str | None = ...): ...
def zeros(shape: Any, dtype: Any | None = ..., name: str | None = ...): ...
def zeros_like(x: Any, dtype: Any | None = ..., name: str | None = ...): ...
def eye(num_rows: Any, num_columns: Any | None = ..., dtype: Any | None = ..., name: str | None = ...): ...
def full(
    shape: ShapeLike,
    fill_value: bool | int | float | Tensor,
    dtype: Any | None = ...,
    name: str | None = ...,
) -> Tensor: ...
def arange(
    start: int = ..., end: Any | None = ..., step: int = ..., dtype: Any | None = ..., name: str | None = ...
): ...
def tril(x: Any, diagonal: int = ..., name: str | None = ...): ...
def triu(x: Any, diagonal: int = ..., name: str | None = ...): ...
def meshgrid(*args: Any, **kwargs: Any): ...
def diagflat(x: Any, offset: int = ..., name: str | None = ...): ...
def diag(x: Any, offset: int = ..., padding_value: int = ..., name: str | None = ...): ...
def empty(shape: Any, dtype: Any | None = ..., name: str | None = ...): ...
def empty_like(x: Any, dtype: Any | None = ..., name: str | None = ...): ...
def assign(x: Any, output: Any | None = ...): ...
def clone(x: Any, name: str | None = ...): ...
def complex(real: Any, imag: Any, name: str | None = ...): ...
