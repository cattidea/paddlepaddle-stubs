from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator, Sized
from typing import Any, SupportsFloat, SupportsIndex, SupportsInt

import numpy.typing as npt
from typing_extensions import TypeAlias

from .basic import Numberic
from .device import Place
from .dtype import DTypeLike, dtype
from .shape import ShapeLike

# `builtins.PyCapsule` unfortunately lacks annotations as of the moment;
# use `Any` as a stopgap measure
# @see also: https://github.com/numpy/numpy/blob/b6a3e837785eac58a2f68e126f4db7895ca047b3/numpy/__init__.pyi#L1465
_PyCapsule: TypeAlias = Any  # noqa: Y047

TensorLike: TypeAlias = list[TensorLike] | tuple[TensorLike, ...] | npt.NDArray[Any] | Tensor | Numberic

class Tensor(Sized, Iterable[Tensor], Hashable, SupportsFloat, SupportsInt, SupportsIndex):
    shape: list[int]
    dtype: dtype
    grad: npt.NDArray[Any]
    is_leaf: bool
    name: str
    ndim: int
    persistable: bool
    place: Place
    stop_gradient: bool
    T: Tensor
    __array_ufunc__: None

    # math operations
    def __neg__(self) -> Tensor: ...
    # def __pos__(self) -> Tensor: ... # missing
    def __add__(self, other: TensorLike) -> Tensor: ...
    def __radd__(self, other: TensorLike) -> Tensor: ...
    def __sub__(self, other: TensorLike) -> Tensor: ...
    def __rsub__(self, other: TensorLike) -> Tensor: ...
    def __mul__(self, other: TensorLike) -> Tensor: ...
    def __rmul__(self, other: TensorLike) -> Tensor: ...
    def __div__(self, other: TensorLike) -> Tensor: ...
    def __rdiv__(self, other: TensorLike) -> Tensor: ...
    def __truediv__(self, other: TensorLike) -> Tensor: ...
    def __rtruediv__(self, other: TensorLike) -> Tensor: ...
    def __floordiv__(self, other: TensorLike) -> Tensor: ...
    def __rfloordiv__(self, other: TensorLike) -> Tensor: ...
    def __mod__(self, other: TensorLike) -> Tensor: ...
    def __rmod__(self, other: TensorLike) -> Tensor: ...
    def __pow__(self, other: TensorLike) -> Tensor: ...
    def __rpow__(self, other: TensorLike) -> Tensor: ...
    def __matmul__(self, other: TensorLike) -> Tensor: ...
    def __rmatmul__(self, other: TensorLike) -> Tensor: ...

    # comparison operations
    def __eq__(self, other: TensorLike) -> Tensor: ...
    def __ne__(self, other: TensorLike) -> Tensor: ...
    def __le__(self, other: TensorLike) -> Tensor: ...
    def __ge__(self, other: TensorLike) -> Tensor: ...
    def __lt__(self, other: TensorLike) -> Tensor: ...
    def __gt__(self, other: TensorLike) -> Tensor: ...

    # bitwise operations
    def __or__(self, other: TensorLike) -> Tensor: ...
    def __ror__(self, other: TensorLike) -> Tensor: ...
    def __and__(self, other: TensorLike) -> Tensor: ...
    def __rand__(self, other: TensorLike) -> Tensor: ...
    def __xor__(self, other: TensorLike) -> Tensor: ...
    def __rxor__(self, other: TensorLike) -> Tensor: ...
    def __invert__(self) -> Tensor: ...
    # def __lshift__(self, other: TensorLike) -> Tensor: ... # missing
    # def __rshift__(self, other: TensorLike) -> Tensor: ... # missing
    # def __rlshift__(self, other: TensorLike) -> Tensor: ... # missing
    # def __rrshift__(self, other: TensorLike) -> Tensor: ... # missing

    # ?: Missing all inplace operations

    # others
    # def __abs__(self) -> Tensor: ... # missing
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Tensor]: ...
    def __getitem__(self, key: int | Tensor | slice) -> Tensor: ...
    def __float__(self) -> float: ...
    def __int__(self) -> int: ...
    def __nonzero__(self) -> bool: ...
    def __index__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __contains__(self, other: TensorLike) -> bool: ...
    # def __dlpack__(self) -> _PyCapsule: ... # missing
    # def __dlpack_device__(self, stream: Any | None = ...) # missing

    # methods
    def clear_grad(self) -> None: ...
    clear_gradient = clear_grad
    def item(self, *args: int) -> Numberic: ...
    def astype(self, dtype: DTypeLike) -> Tensor: ...
    def reshape(self, shape: ShapeLike, name: str | None = ...) -> Tensor: ...
    def set_value(self, value: TensorLike) -> None: ...
    def numpy(self) -> npt.NDArray[Any]: ...
    def backward(
        self,
        grad_tensor: Tensor | None = ...,
        retain_graph: bool = ...,
    ) -> None: ...
    def clone(self) -> Tensor: ...
    def broadcast_to(self, shape: ShapeLike, name: str | None = ...) -> Tensor: ...
    def cast(self, dtype: DTypeLike) -> Tensor: ...
    @property
    def size(self) -> int: ...
    def dim(self) -> int: ...
    def ndimension(self) -> int: ...
    def imag(self) -> Tensor: ...
    def real(self) -> Tensor: ...

    # extended math ops
    # TODO: ...
