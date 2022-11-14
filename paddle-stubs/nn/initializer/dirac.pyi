from __future__ import annotations

from typing import Any, Optional

from ..._typing import Tensor
from ...fluid.initializer import Initializer

class Dirac(Initializer):
    def __init__(self, groups: int = ..., name: str | None = ...) -> None: ...
    def __call__(self, var: Tensor, block: Any | None = ...) -> Tensor: ...
