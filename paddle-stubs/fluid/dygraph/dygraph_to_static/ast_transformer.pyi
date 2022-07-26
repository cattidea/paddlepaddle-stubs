from __future__ import annotations

from typing import Any

from paddle.utils import gast

class DygraphToStaticAst(gast.NodeTransformer):
    translator_logger: Any = ...
    def __init__(self) -> None: ...
    root: Any = ...
    static_analysis_visitor: Any = ...
    static_analysis_root: Any = ...
    decorate_func_name: Any = ...
    def get_static_ast(self, root: Any): ...
    def transfer_from_node_type(self, node_wrapper: Any) -> None: ...
    def visit_FunctionDef(self, node: Any): ...
    def get_module_name(self): ...
