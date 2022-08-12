from __future__ import annotations

from typing import Any, Optional, TypeVar

import numpy as np
from typing_extensions import Literal, Self

# Scalar

Numberic = int | float | complex | np.number[Any]

# Scalar Sequence

_T = TypeVar("_T", bound=Numberic)
_SeqLevel1 = tuple[_T, ...] | list[_T]

_TL1 = TypeVar("_TL1", bound=_SeqLevel1[Numberic])
_SeqLevel2 = tuple[_TL1, ...] | list[_TL1]

_TL2 = TypeVar("_TL2", bound=_SeqLevel2[_SeqLevel1[Numberic]])
_SeqLevel3 = tuple[_TL2, ...] | list[_TL2]

_TL3 = TypeVar("_TL3", bound=_SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]])
_SeqLevel4 = tuple[_TL3, ...] | list[_TL3]

_TL4 = TypeVar("_TL4", bound=_SeqLevel4[_SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]]])
_SeqLevel5 = tuple[_TL4, ...] | list[_TL4]

_TL5 = TypeVar("_TL5", bound=_SeqLevel5[_SeqLevel4[_SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]]]])
_SeqLevel6 = tuple[_TL5, ...] | list[_TL5]

IntSequence = _SeqLevel1[int]
NumbericSequence = _SeqLevel1[Numberic]
NestedNumbericSequence = (
    Numberic
    | _SeqLevel1[Numberic]
    | _SeqLevel2[_SeqLevel1[Numberic]]
    | _SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]]
    | _SeqLevel4[_SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]]]
    | _SeqLevel5[_SeqLevel4[_SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]]]]
    | _SeqLevel6[_SeqLevel5[_SeqLevel4[_SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]]]]]
)

# Shape

ShapeLike = tuple[int, ...] | list[int] | Tensor

# DType

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

# Tensor

class Tensor:
    shape: list[int]
    dtype: dtype
    def __add__(self, other: Self | np.ndarray[Any, Any] | Numberic) -> Tensor: ...
    def __radd__(self, other: Self | np.ndarray[Any, Any] | Numberic) -> Tensor: ...
    def __sub__(self, other: Self | np.ndarray[Any, Any] | Numberic) -> Tensor: ...
    def __rsub__(self, other: Self | np.ndarray[Any, Any] | Numberic) -> Tensor: ...
    def __mul__(self, other: Self | np.ndarray[Any, Any] | Numberic) -> Tensor: ...
    def __rmul__(self, other: Self | np.ndarray[Any, Any] | Numberic) -> Tensor: ...
    def __div__(self, other: Self | np.ndarray[Any, Any] | Numberic) -> Tensor: ...
    def __rdiv__(self, other: Self | np.ndarray[Any, Any] | Numberic) -> Tensor: ...
    def __truediv__(self, other: Self | np.ndarray[Any, Any] | Numberic) -> Tensor: ...
    def __rtruediv__(self, other: Self | np.ndarray[Any, Any] | Numberic) -> Tensor: ...
    def __pow__(self, other: Self | np.ndarray[Any, Any] | Numberic) -> Tensor: ...
    def __rpow__(self, other: Self | np.ndarray[Any, Any] | Numberic) -> Tensor: ...
    def astype(self, dtype: DTypeLike) -> Tensor: ...
    def reshape(self, shape: ShapeLike, name: Optional[str] = ...) -> Tensor: ...
    def set_value(self, value: Tensor | np.ndarray[Any, Any]) -> None: ...
    def numpy(self) -> np.ndarray[Any, Any]: ...

# Device

class Place: ...
class CPUPlace(Place): ...

class CUDAPlace(Place):
    def __init__(self, id: int) -> None: ...

class CUDAPinnedPlace(Place): ...

class NPUPlace(Place):
    def __init__(self, id: int) -> None: ...

class IPUPlace(Place): ...

class CustomPlace(Place):
    def __init__(self, name: str, id: int) -> None: ...

class MLUPlace(Place):
    def __init__(self, id: int) -> None: ...

class XPUPlace(Place):
    def __init__(self, id: int) -> None: ...

PlaceLike = (
    CPUPlace | CUDAPlace | CUDAPinnedPlace | NPUPlace | IPUPlace | CustomPlace | MLUPlace | XPUPlace | str
)  # TODO: only support the literal like "dev:id"

# Layout

DataLayout0D = Literal["NC"]
DataLayout1D = Literal["NCL", "NLC"]
DataLayout2D = Literal["NCHW", "NHCW"]
DataLayout3D = Literal["NCDHW", "NDHWC"]
DataLayoutND = DataLayout0D | DataLayout1D | DataLayout2D | DataLayout3D

DataLayout1DVariant = Literal["NCW", "NWC"]
DataLayoutImage = Literal["HWC", "CHW"]
