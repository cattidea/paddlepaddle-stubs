from __future__ import annotations

from typing import Callable, Optional

from typing_extensions import ParamSpec, TypeVar

from .._typing import Tensor
from .backward_mode import backward as backward
from .py_layer import PyLayer as PyLayer
from .py_layer import PyLayerContext as PyLayerContext

_InputArgs = ParamSpec("_InputArgs")
_RetValue = TypeVar("_RetValue")

def grad(
    outputs: Tensor | list[Tensor] | tuple[Tensor],
    inputs: Tensor | list[Tensor] | tuple[Tensor],
    grad_outputs: Tensor | list[Tensor | None] | tuple[Tensor | None, ...] | None = ...,
    retain_graph: bool = ...,
    create_graph: bool = ...,
    only_inputs: bool = ...,
    allow_unused: bool = ...,
    no_grad_vars: Tensor | list[Tensor | None] | tuple[Tensor | None, ...] | None = None,
): ...
def no_grad(func: Callable[_InputArgs, _RetValue]) -> Callable[_InputArgs, _RetValue]: ...

class EagerPyLayer: ...

def is_grad_enabled() -> bool: ...
def set_grad_enabled(mode: bool) -> None: ...
