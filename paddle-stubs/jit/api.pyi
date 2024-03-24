from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from typing_extensions import Literal, ParamSpec, TypeAlias

from paddle.static import BuildStrategy, InputSpec

_RetT = TypeVar("_RetT")
_InputT = ParamSpec("_InputT")
Backends: TypeAlias = Literal["CINN"]

def to_static(
    function: Callable[_InputT, _RetT],
    input_spec: InputSpec | None = ...,
    build_strategy: BuildStrategy | None = ...,
    backend: Backends | None = ...,
    **kwargs: Any,
) -> Callable[_InputT, _RetT]: ...
def not_to_static(func: Callable[_InputT, _RetT] | None = None) -> Callable[_InputT, _RetT]: ...
