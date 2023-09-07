from __future__ import annotations

from typing import Any

from paddle.base.dygraph.layers import Layer

class MetaParallelBase(Layer):
    def __init__(self, layers: Any, hcg: Any, strategy: Any) -> None: ...
    def forward(self, *inputs: Any, **kwargs: Any): ...
