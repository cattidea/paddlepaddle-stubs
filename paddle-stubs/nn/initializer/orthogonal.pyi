from __future__ import annotations

from typing import Any, Optional

from ..._typing import Tensor
from ...base.initializer import Initializer

class Orthogonal(Initializer):
    def __init__(self, gain: float = ..., name: str | None = ...) -> None: ...
    def __call__(self, var: Tensor, block: Any | None = ...) -> Tensor: ...
