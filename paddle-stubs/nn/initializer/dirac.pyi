from __future__ import annotations

from typing import Any, Optional

from ..._typing import Tensor
from ...fluid.initializer import Initializer

class Dirac(Initializer):
    def __init__(self, groups: int = ..., name: Optional[str] = ...) -> None: ...
    def __call__(self, var: Tensor, block: Optional[Any] = ...) -> Tensor: ...
