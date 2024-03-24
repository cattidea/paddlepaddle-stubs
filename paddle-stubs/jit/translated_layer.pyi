from __future__ import annotations

from typing import Any

from paddle.base.dygraph import layers
from paddle.static import Program

class _ProgramHolder:
    def __init__(self, program_desc: Any) -> None: ...
    @property
    def infer_program(self) -> Program: ...
    @property
    def train_program(self) -> Program: ...
    @property
    def input_descs(self): ...
    @property
    def output_descs(self): ...
    @property
    def persistable_names(self): ...
    @property
    def double_grad_descs(self): ...
    @property
    def scope(self): ...

class TranslatedLayer(layers.Layer):
    def __init__(self, programs: Any, persistable_vars: Any) -> None: ...
    training: bool = ...
    def train(self) -> None: ...
    def eval(self) -> None: ...
    def program(self, method_name: str = ...) -> Program: ...