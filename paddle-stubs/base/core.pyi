from __future__ import annotations

from collections.abc import Callable
from types import CodeType, FrameType
from typing import Any, NamedTuple

class CustomCode(NamedTuple):
    code: CodeType
    disable_eval_frame: bool

CallbackFunc = Callable[[FrameType], CustomCode | None]

core_suffix: str
has_avx_core: bool
has_noavx_core: bool
current_path: Any
third_lib_path: Any
executable_path: Any
VarBase = Any
VarDesc = Any
eager = Any

def avx_supported(): ...
def run_shell_command(cmd: Any): ...
def get_dso_path(core_so: Any, dso_name: Any): ...
def load_dso(dso_absolute_path: Any) -> None: ...
def pre_load(dso_name: Any) -> None: ...
def get_libc_ver(): ...
def less_than_ver(a: Any, b: Any): ...

libc_type: Any
libc_ver: Any
load_noavx: bool

def set_paddle_custom_device_lib_path(lib_path: Any) -> None: ...
def set_paddle_lib_path() -> None: ...
def set_eval_frame(callback: CallbackFunc | None) -> CallbackFunc | None: ...

class TracerEventType: ...
class Tracer: ...
class Generator: ...