from __future__ import annotations

from typing import Any

from paddle.utils import gast

class BasicApiTransformer(gast.NodeTransformer):
    wrapper_root: Any = ...
    root: Any = ...
    class_node_dict: Any = ...
    def __init__(self, wrapper_root: Any) -> None: ...
    def transform(self): ...
    def visit_Assign(self, node: Any): ...
    def visit_Expr(self, node: Any): ...

class ToTensorTransformer(gast.NodeTransformer):
    root: Any = ...
    def __init__(self, node: Any) -> None: ...
    def transform(self): ...
    def visit_Call(self, node: Any): ...

def is_to_variable(node: Any): ...
def to_assign_node(node: Any): ...
