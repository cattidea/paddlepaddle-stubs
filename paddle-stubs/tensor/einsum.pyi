from __future__ import annotations

from .._typing import Tensor

def einsum(equation: str, *operands: Tensor) -> Tensor: ...
