from __future__ import annotations

from typing import Callable, Optional

from typing_extensions import ParamSpec, TypeVar

from .._typing import Tensor
from .backward_mode import backward as backward
from .py_layer import PyLayer as PyLayer
from .py_layer import PyLayerContext as PyLayerContext

InputArgs = ParamSpec("InputArgs")
RetValue = TypeVar("RetValue")

def grad(
    outputs: Tensor | list[Tensor] | tuple[Tensor],
    inputs: Tensor | list[Tensor] | tuple[Tensor],
    grad_outputs: Optional[Tensor | list[Optional[Tensor]] | tuple[Optional[Tensor], ...]] = ...,
    retain_graph: bool = ...,
    create_graph: bool = ...,
    only_inputs: bool = ...,
    allow_unused: bool = ...,
    no_grad_vars: Optional[Tensor | list[Optional[Tensor]] | tuple[Optional[Tensor], ...]] = None,
): ...
def no_grad(func: Callable[InputArgs, RetValue]) -> Callable[InputArgs, RetValue]: ...

class EagerPyLayer: ...

def is_grad_enabled() -> bool: ...
def set_grad_enabled(mode: bool) -> None: ...
