from __future__ import annotations

from typing_extensions import TypedDict

from .. import nn
from .._typing import DTypeLike, DynamicShapeLike, Tensor

class ModelSummary(TypedDict):
    total_params: int
    trainable_params: int

def summary(
    net: nn.Layer,
    input_size: list[DynamicShapeLike] | None = ...,
    dtypes: DTypeLike | None = ...,
    input: Tensor | None = ...,
) -> ModelSummary: ...
