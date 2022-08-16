from __future__ import annotations

from collections.abc import Iterable, Iterator, Sized
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from .basic import Numberic
from .dtype import DTypeLike, dtype
from .shape import ShapeLike

TensorLike = list["TensorLike"] | tuple["TensorLike", ...] | npt.NDArray[Any] | Tensor | Numberic

class Tensor(Sized, Iterable[Tensor]):
    shape: list[int]
    dtype: dtype
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
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Tensor]: ...
    def __getitem__(self, key: int | Tensor | slice) -> Tensor: ...
    def astype(self, dtype: DTypeLike) -> Tensor: ...
    def reshape(self, shape: ShapeLike, name: Optional[str] = ...) -> Tensor: ...
    def set_value(self, value: TensorLike) -> None: ...
    def numpy(self) -> npt.NDArray[Any]: ...
    def backward(
        self,
        grad_tensor: Optional[Tensor] = ...,
        retain_graph: bool = ...,
    ) -> None: ...
