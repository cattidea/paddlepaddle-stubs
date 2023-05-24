from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from paddle.static import BuildStrategy, InputSpec
from typing_extensions import Literal, ParamSpec

RetT = TypeVar("RetT")
InputT = ParamSpec("InputT")
Backends = Literal["CINN"]

def to_static(
    function: Callable[InputT, RetT],
    input_spec: InputSpec | None = ...,
    build_strategy: BuildStrategy | None = ...,
    backend: Backends | None = ...,
    **kwargs: Any,
) -> Callable[InputT, RetT]: ...
