from __future__ import annotations

from typing import Any, Optional

from .base_cost import Cost as Cost

class TensorCost:
    def __init__(
        self,
        tensor: Any | None = ...,
        dist_tensor: Any | None = ...,
        shape: Any | None = ...,
        dtype: Any | None = ...,
    ) -> None: ...
    @property
    def tensor(self): ...
    @property
    def dist_tensor(self): ...
    @property
    def shape(self): ...
    @property
    def dtype(self): ...
    @property
    def cost(self): ...
    def calc_cost(self): ...
