from __future__ import annotations

from typing import Any, Optional

from paddle.fluid.data_feeder import check_dtype as check_dtype
from paddle.fluid.data_feeder import (
    check_variable_and_dtype as check_variable_and_dtype,
)

from ..framework import core as core

class PrintOptions:
    precision: int = ...
    threshold: int = ...
    edgeitems: int = ...
    linewidth: int = ...
    sci_mode: bool = ...

DEFAULT_PRINT_OPTIONS: Any

def set_printoptions(
    precision: Optional[Any] = ...,
    threshold: Optional[Any] = ...,
    edgeitems: Optional[Any] = ...,
    sci_mode: Optional[Any] = ...,
    linewidth: Optional[Any] = ...,
) -> None: ...
def to_string(var: Any, prefix: str = ...): ...
def sparse_tensor_to_string(tensor: Any, prefix: str = ...): ...
def tensor_to_string(tensor: Any, prefix: str = ...): ...
