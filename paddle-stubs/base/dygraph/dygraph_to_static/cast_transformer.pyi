from __future__ import annotations

from typing import Any

from paddle.utils import gast

class CastTransformer(gast.NodeTransformer):
    def __init__(self, wrapper_root: Any) -> None: ...
    def transform(self) -> None: ...
    def visit_Call(self, node: Any): ...
