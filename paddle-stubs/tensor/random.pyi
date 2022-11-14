from __future__ import annotations

from typing import Any, Optional

from .. import Tensor
from .._typing import DTypeLike, ShapeLike

def bernoulli(x: Any, name: str | None = ...): ...
def poisson(x: Any, name: str | None = ...): ...
def multinomial(x: Any, num_samples: int = ..., replacement: bool = ..., name: str | None = ...): ...
def gaussian(
    shape: Any, mean: float = ..., std: float = ..., dtype: Any | None = ..., name: str | None = ...
): ...
def standard_normal(shape: Any, dtype: Any | None = ..., name: str | None = ...): ...
def randn(
    shape: Any,
    dtype: DTypeLike | None = ...,
    name: str | None = ...,
) -> Tensor: ...
def normal(mean: float = ..., std: float = ..., shape: Any | None = ..., name: str | None = ...): ...
def uniform(
    shape: ShapeLike,
    dtype: DTypeLike | None = ...,
    min: float = ...,
    max: float = ...,
    seed: int = ...,
    name: str | None = ...,
) -> Tensor: ...
def uniform_(
    x: Tensor,
    min: float = ...,
    max: float = ...,
    seed: int = ...,
    name: str | None = ...,
) -> Tensor: ...
def randint(
    low: int = ...,
    high: int | None = ...,
    shape: ShapeLike = ...,
    dtype: DTypeLike | None = ...,
    name: str | None = ...,
) -> Tensor: ...
def randint_like(
    x: Tensor,
    low: int = ...,
    high: int | None = ...,
    dtype: DTypeLike | None = ...,
    name: str | None = ...,
) -> Tensor: ...
def randperm(n: Any, dtype: str = ..., name: str | None = ...): ...
def rand(
    shape: ShapeLike,
    dtype: DTypeLike | None = ...,
    name: str | None = ...,
) -> Tensor: ...
def exponential_(x: Any, lam: float = ..., name: str | None = ...): ...
