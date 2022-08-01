from __future__ import annotations

from typing import Any

from ... import Tensor, nn

class LeNet(nn.Layer):
    def __init__(self, num_classes: int = ...) -> None: ...
    def forward(self, inputs: Any) -> Tensor: ...
