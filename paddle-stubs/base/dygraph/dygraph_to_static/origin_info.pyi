from __future__ import annotations

from typing import Any, Optional

from paddle.utils import gast

ORIGI_INFO: str
ORIGI_INFO_MAP: str

class Location:
    filepath: Any = ...
    lineno: Any = ...
    col_offset: Any = ...
    def __init__(self, filepath: Any, lineno: Any, col_offset: Any | None = ...) -> None: ...
    @property
    def line_location(self): ...

class OriginInfo:
    location: Any = ...
    function_name: Any = ...
    source_code: Any = ...
    def __init__(self, location: Any, function_name: Any, source_code: Any) -> None: ...
    def formated_message(self): ...
    def as_frame(self): ...

class OriginInfoAttacher(gast.NodeTransformer):
    root: Any = ...
    func: Any = ...
    filepath: Any = ...
    source_code: Any = ...
    current_func: Any = ...
    def __init__(self, root: Any, func: Any) -> None: ...
    col_offset: Any = ...
    source_lines: Any = ...
    lineno_offset: Any = ...
    def transform(self) -> None: ...
    def visit(self, node: Any): ...

global_origin_info_map: Any

def create_and_update_origin_info_map(transformed_node: Any, static_func: Any, is_global: bool = ...): ...
def attach_origin_info(ast_node: Any, func: Any): ...
def ast_walk(transformed_node: Any, static_node: Any): ...
def update_op_callstack_with_origin_info(program: Any): ...