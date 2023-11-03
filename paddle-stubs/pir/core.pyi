from __future__ import annotations

from typing import Any

import numpy as np

from paddle import Tensor
from paddle._typing import DTypeLike, ShapeLike
from paddle.base.core import VarDesc
from paddle.base.libpaddle import DataType

from ..base.wrapped_decorator import signature_safe_contextmanager

vartype_to_datatype: dict[VarDesc.VarType, DataType]

np_type_to_paddle_type: dict[str, DataType]

def convert_np_dtype_to_dtype_(np_dtype: str | np.dtype) -> DataType: ...
def default_startup_program(): ...
def default_main_program(): ...
def switch_main_program(program): ...
def switch_startup_program(program): ...
@signature_safe_contextmanager
def program_guard(main_program, startup_program=None): ...

class ParameterMeta:
    shape: ShapeLike
    dtype: DTypeLike

def create_parameter(
    dtype: DTypeLike,
    shape: ShapeLike,
    **kwargs: Any,
): ...
def _convert_into_opresult(tensor: Tensor) -> None: ...
