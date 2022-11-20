from __future__ import annotations

from typing_extensions import TypeAlias

from .tensor import Tensor

ShapeLike: TypeAlias = tuple[int, ...] | list[int] | Tensor
DynamicShapeLike: TypeAlias = tuple[None | int, ...] | list[None | int] | Tensor
