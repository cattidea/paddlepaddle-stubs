from __future__ import annotations

from typing import Any

def crepr(v: Any): ...

class Rank:
    kind: Any = ...
    name: Any = ...
    priority: Any = ...
    nodes: Any = ...
    def __init__(self, kind: Any, name: Any, priority: Any) -> None: ...

class Graph:
    rank_counter: int = ...
    title: Any = ...
    attrs: Any = ...
    nodes: Any = ...
    edges: Any = ...
    rank_groups: Any = ...
    def __init__(self, title: Any, **attrs: Any) -> None: ...
    def code(self): ...
    def rank_group(self, kind: Any, priority: Any): ...
    def node(self, label: Any, prefix: Any, description: str = ..., **attrs: Any): ...
    def edge(self, source: Any, target: Any, **attrs: Any): ...
    def compile(self, dot_path: Any): ...
    def show(self, dot_path: Any) -> None: ...

class Node:
    counter: int = ...
    label: Any = ...
    name: Any = ...
    description: Any = ...
    attrs: Any = ...
    def __init__(self, label: Any, prefix: Any, description: str = ..., **attrs: Any) -> None: ...

class Edge:
    source: Any = ...
    target: Any = ...
    attrs: Any = ...
    def __init__(self, source: Any, target: Any, **attrs: Any) -> None: ...

class GraphPreviewGenerator:
    graph: Any = ...
    op_rank: Any = ...
    param_rank: Any = ...
    arg_rank: Any = ...
    def __init__(self, title: Any) -> None: ...
    def __call__(self, path: str = ..., show: bool = ...) -> None: ...
    def add_param(self, name: Any, data_type: Any, highlight: bool = ...): ...
    def add_op(self, opType: Any, **kwargs: Any): ...
    def add_arg(self, name: Any, highlight: bool = ...): ...
    def add_edge(self, source: Any, target: Any, **kwargs: Any): ...
