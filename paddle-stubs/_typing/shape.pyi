from __future__ import annotations

from .tensor import Tensor

ShapeLike = tuple[int, ...] | list[int] | Tensor
DynamicShapeLike = tuple[None | int, ...] | list[None | int] | Tensor
