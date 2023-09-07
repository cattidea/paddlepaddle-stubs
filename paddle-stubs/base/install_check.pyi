from __future__ import annotations

from typing import Any

from .dygraph import Layer

class SimpleLayer(Layer):
    def __init__(self, input_size: Any) -> None: ...
    def forward(self, inputs: Any): ...

def run_check(): ...
