from __future__ import annotations

from typing import Any

from paddle.utils import gast

class ListTransformer(gast.NodeTransformer):
    wrapper_root: Any = ...
    root: Any = ...
    list_name_to_updated: Any = ...
    list_nodes: Any = ...
    static_analysis_visitor: Any = ...
    node_to_wrapper_map: Any = ...
    scope_var_type_dict: Any = ...
    def __init__(self, wrapper_root: Any) -> None: ...
    def transform(self) -> None: ...
    def visit_Call(self, node: Any): ...
    def visit_Assign(self, node: Any): ...
    def visit_If(self, node: Any): ...
    def visit_While(self, node: Any): ...
    def visit_For(self, node: Any): ...
    def replace_list_with_tensor_array(self, node: Any) -> None: ...
