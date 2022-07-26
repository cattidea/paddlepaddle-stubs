from __future__ import annotations

from typing import Any, Optional

from paddle.nn import Layer

class ReLU(Layer):
    def __init__(self, name: Optional[Any] = ...) -> None: ...
    def forward(self, x: Any): ...
    def extra_repr(self): ...
