from __future__ import annotations

from collections import defaultdict as defaultdict
from typing import Any

from .graphviz import Graph as Graph

logger: Any
OP_STYLE: Any
VAR_STYLE: Any
GRAPH_STYLE: Any
GRAPH_ID: int

def unique_id(): ...
def draw_node(op: Any): ...
def draw_edge(var_parent: Any, op: Any, var: Any, arg: Any): ...
def parse_graph(program: Any, graph: Any, var_dict: Any, **kwargs: Any) -> None: ...
def draw_graph(startup_program: Any, main_program: Any, **kwargs: Any): ...
