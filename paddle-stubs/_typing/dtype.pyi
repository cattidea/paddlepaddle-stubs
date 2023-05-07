from __future__ import annotations

from typing import Any

import numpy as np
from typing_extensions import Literal, TypeAlias

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

_DTypeString: TypeAlias = Literal[
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

_DTypeNumpy: TypeAlias = (
    type[
        np.uint8
        | np.int8
        | np.int16
        | np.int32
        | np.int64
        | np.float32
        | np.float64
        | np.float16
        | np.complex64
        | np.complex128
        | np.bool_
    ]
    | np.dtype[Any]
)

DTypeLike: TypeAlias = dtype | _DTypeNumpy | _DTypeString | None
