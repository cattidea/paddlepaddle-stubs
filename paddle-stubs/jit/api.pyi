from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any, TypedDict, TypeVar

from typing_extensions import Literal, ParamSpec, TypeAlias, Unpack

from paddle.static import BuildStrategy, InputSpec

from .translated_layer import TranslatedLayer

_RetT = TypeVar("_RetT")
_InputT = ParamSpec("_InputT")
Backends: TypeAlias = Literal["CINN"]

class _SaveLoadConfig(TypedDict):
    output_spec: Any
    with_hook: Any
    combine_params: Any
    clip_extra: Any
    skip_forward: Any
    input_names_after_prune: Any
    skip_prune_program: Any

def to_static(
    function: Callable[_InputT, _RetT],
    input_spec: InputSpec | None = ...,
    build_strategy: BuildStrategy | None = ...,
    backend: Backends | None = ...,
    **kwargs: Any,
) -> Callable[_InputT, _RetT]: ...
def not_to_static(func: Callable[_InputT, _RetT] | None = None) -> Callable[_InputT, _RetT]: ...
def enable_to_static(enable_to_static_bool: bool) -> None: ...
def ignore_module(modules: list[ModuleType]) -> None: ...
def set_code_level(level: int = 100, also_to_stdout: bool = False) -> None: ...
def set_verbosity(level: int = 0, also_to_stdout: bool = False) -> None: ...
def save(
    layer: Callable[_InputT, _RetT],
    path: str,
    input_spec: InputSpec | None = None,
    **configs: Unpack[_SaveLoadConfig],
) -> None: ...
def load(path: str, **configs: Unpack[_SaveLoadConfig]) -> TranslatedLayer: ...
