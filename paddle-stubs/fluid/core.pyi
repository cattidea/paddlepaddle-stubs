from __future__ import annotations

from typing import Any

from . import core_avx as core_avx
from . import core_noavx as core_noavx
from .core_avx import *
from .core_noavx import *
from .core_noavx import __doc__ as __doc__
from .core_noavx import __file__ as __file__
from .core_noavx import __name__ as __name__
from .core_noavx import __package__ as __package__
from .core_noavx import __unittest_throw_exception__ as __unittest_throw_exception__

core_suffix: str
has_avx_core: bool
has_noavx_core: bool
current_path: Any
third_lib_path: Any
executable_path: Any

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
