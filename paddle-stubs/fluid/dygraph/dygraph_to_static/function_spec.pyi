from __future__ import annotations

from typing import Any, Optional

class FunctionSpec:
    varargs_name: Any = ...
    def __init__(self, function: Any, input_spec: Any | None = ...) -> None: ...
    def unified_args_and_kwargs(self, args: Any, kwargs: Any): ...
    def args_to_input_spec(self, args: Any, kwargs: Any): ...
    def to_static_inputs_with_spec(self, input_with_spec: Any, main_program: Any): ...
    @property
    def dygraph_function(self): ...
    @property
    def args_name(self): ...
    @property
    def input_spec(self): ...
    @property
    def flat_input_spec(self): ...
    @property
    def code(self): ...

def get_parameters(layer_instance: Any, include_sublayer: bool = ...): ...
def get_buffers(layer_instance: Any, include_sublayer: bool = ...): ...
def convert_to_input_spec(inputs: Any, input_spec: Any): ...
def replace_spec_empty_name(args_name: Any, input_with_spec: Any): ...
