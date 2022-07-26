from __future__ import annotations

from ..._typing import DataLayout2D, Tensor
from .. import Layer as Layer
from .. import functional as functional

class PixelShuffle(Layer):
    def __init__(
        self,
        upscale_factor: int,
        data_format: DataLayout2D = ...,
        name: str | None = ...,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    __call__ = forward
