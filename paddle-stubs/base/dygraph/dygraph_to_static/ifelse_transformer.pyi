from __future__ import annotations

from typing import Any, Optional

from paddle.utils import gast

TRUE_FUNC_PREFIX: str
FALSE_FUNC_PREFIX: str

class IfElseTransformer(gast.NodeTransformer):
    root: Any = ...
    static_analysis_visitor: Any = ...
    def __init__(self, wrapper_root: Any) -> None: ...
    def transform(self) -> None: ...
    def visit_If(self, node: Any): ...
    def visit_Call(self, node: Any): ...
    def visit_IfExp(self, node: Any): ...

class NameVisitor(gast.NodeVisitor):
    after_node: Any = ...
    end_node: Any = ...
    name_ids: Any = ...
    ancestor_nodes: Any = ...
    def __init__(self, after_node: Any | None = ..., end_node: Any | None = ...) -> None: ...
    def visit(self, node: Any): ...
    def visit_If(self, node: Any) -> None: ...
    def visit_Attribute(self, node: Any) -> None: ...
    def visit_Name(self, node: Any) -> None: ...
    def visit_Assign(self, node: Any) -> None: ...
    def visit_FunctionDef(self, node: Any) -> None: ...

def get_name_ids(nodes: Any, after_node: Any | None = ..., end_node: Any | None = ...): ...
def parse_cond_args(parent_ids_dict: Any, var_ids_dict: Any, modified_ids_dict: Any | None = ..., ctx: Any = ...): ...
def parse_cond_return(parent_vars_dict: Any, if_vars_dict: Any, else_vars_dict: Any, after_ifelse_vars_dict: Any): ...
def transform_if_else(node: Any, root: Any): ...
def create_convert_ifelse_node(
    return_name_ids: Any, pred: Any, true_func: Any, false_func: Any, is_if_expr: bool = ...
): ...