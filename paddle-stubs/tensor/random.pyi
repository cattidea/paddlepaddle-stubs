from __future__ import annotations

from typing import Any, Optional

from .. import Tensor
from .._typing import DTypeLike, ShapeLike

def bernoulli(x: Any, name: Optional[Any] = ...): ...
def poisson(x: Any, name: Optional[Any] = ...): ...
def multinomial(x: Any, num_samples: int = ..., replacement: bool = ..., name: Optional[Any] = ...): ...
def gaussian(
    shape: Any, mean: float = ..., std: float = ..., dtype: Optional[Any] = ..., name: Optional[Any] = ...
): ...
def standard_normal(shape: Any, dtype: Optional[Any] = ..., name: Optional[Any] = ...): ...
def randn(
    shape: Any,
    dtype: Optional[DTypeLike] = ...,
    name: Optional[Any] = ...,
) -> Tensor: ...
def normal(mean: float = ..., std: float = ..., shape: Optional[Any] = ..., name: Optional[Any] = ...): ...
def uniform(
    shape: ShapeLike,
    dtype: Optional[DTypeLike] = ...,
    min: float = ...,
    max: float = ...,
    seed: int = ...,
    name: Optional[str] = ...,
) -> Tensor: ...
def uniform_(
    x: Tensor,
    min: float = ...,
    max: float = ...,
    seed: int = ...,
    name: Optional[str] = ...,
) -> Tensor: ...
def randint(
    low: int = ...,
    high: Optional[int] = ...,
    shape: ShapeLike = ...,
    dtype: Optional[DTypeLike] = ...,
    name: Optional[str] = ...,
) -> Tensor: ...
def randint_like(
    x: Tensor,
    low: int = ...,
    high: Optional[int] = ...,
    dtype: Optional[DTypeLike] = ...,
    name: Optional[str] = ...,
) -> Tensor: ...
def randperm(n: Any, dtype: str = ..., name: Optional[Any] = ...): ...
def rand(
    shape: ShapeLike,
    dtype: Optional[DTypeLike] = ...,
    name: Optional[str] = ...,
) -> Tensor: ...
def exponential_(x: Any, lam: float = ..., name: Optional[Any] = ...): ...
