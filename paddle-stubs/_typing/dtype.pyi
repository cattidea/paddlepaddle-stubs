from __future__ import annotations

from typing import Any, Optional

import numpy as np
from typing_extensions import Literal

class dtype:
    def __init__(self, arg0: int) -> None: ...

uint8: dtype
int8: dtype
int16: dtype
int32: dtype
int64: dtype
float32: dtype
float64: dtype
float16: dtype
bfloat16: dtype
complex64: dtype
complex128: dtype
bool: dtype

_DTypeString = Literal[
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float32",
    "float64",
    "float16",
    "bfloat16",
    "complex64",
    "complex128",
    "bool",
]

_DTypeNumpy = (
    type[np.uint8]
    | type[np.int8]
    | type[np.int16]
    | type[np.int32]
    | type[np.int64]
    | type[np.float32]
    | type[np.float64]
    | type[np.float16]
    | type[np.complex64]
    | type[np.complex128]
    | type[np.bool_]
    | np.dtype[Any]
)

DTypeLike = Optional[dtype | _DTypeNumpy | _DTypeString]
