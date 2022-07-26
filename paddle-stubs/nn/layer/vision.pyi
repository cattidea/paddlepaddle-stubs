from __future__ import annotations

from typing import Any, Optional

from .. import Layer as Layer
from .. import functional as functional

class PixelShuffle(Layer):
    def __init__(self, upscale_factor: Any, data_format: str = ..., name: Optional[Any] = ...) -> None: ...
    def forward(self, x: Any): ...
    def extra_repr(self): ...
