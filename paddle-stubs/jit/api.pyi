from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from paddle.static import BuildStrategy, InputSpec
from typing_extensions import Literal, ParamSpec

_RetT = TypeVar("_RetT")
_InputT = ParamSpec("_InputT")
Backends = Literal["CINN"]

def to_static(
    function: Callable[_InputT, _RetT],
    input_spec: InputSpec | None = ...,
    build_strategy: BuildStrategy | None = ...,
    backend: Backends | None = ...,
    **kwargs: Any,
) -> Callable[_InputT, _RetT]: ...
