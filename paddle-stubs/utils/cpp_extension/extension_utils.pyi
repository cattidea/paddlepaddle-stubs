from __future__ import annotations

from collections import namedtuple
from typing import Any, Optional

from ...fluid import core as core
from ...fluid.framework import OpProtoHolder as OpProtoHolder
from ...sysconfig import get_include as get_include
from ...sysconfig import get_lib as get_lib

logger: Any
formatter: Any
ch: Any
OS_NAME: Any
IS_WINDOWS: Any
MSVC_COMPILE_FLAGS: Any
CLANG_COMPILE_FLAGS: Any
CLANG_LINK_FLAGS: Any
MSVC_LINK_FLAGS: Any
COMMON_HIPCC_FLAGS: Any
COMMON_NVCC_FLAGS: Any
GCC_MINI_VERSION: Any
MSVC_MINI_VERSION: Any
WRONG_COMPILER_WARNING: str
ABI_INCOMPATIBILITY_WARNING: str
DEFAULT_OP_ATTR_NAMES: Any

def bootstrap_context() -> None: ...
def load_op_meta_info_and_register_op(lib_filename: Any): ...
def custom_write_stub(resource: Any, pyfile: Any) -> None: ...

OpInfo = namedtuple("OpInfo", ["so_name", "so_path"])

class CustomOpInfo:
    @classmethod
    def instance(cls): ...
    op_info_map: Any = ...
    def __init__(self) -> None: ...
    def add(self, op_name: Any, so_name: Any, so_path: Optional[Any] = ...) -> None: ...
    def last(self): ...

VersionFields = namedtuple(
    "VersionFields",
    [
        "sources",
        "extra_compile_args",
        "extra_link_args",
        "library_dirs",
        "runtime_library_dirs",
        "include_dirs",
        "define_macros",
        "undef_macros",
    ],
)

class VersionManager:
    version_field: Any = ...
    version: Any = ...
    def __init__(self, version_field: Any) -> None: ...
    def hasher(self, version_field: Any): ...
    @property
    def details(self): ...

def combine_hash(md5: Any, value: Any): ...
def clean_object_if_change_cflags(so_path: Any, extension: Any): ...
def prepare_unix_cudaflags(cflags: Any): ...
def prepare_win_cudaflags(cflags: Any): ...
def add_std_without_repeat(cflags: Any, compiler_type: Any, use_std14: bool = ...) -> None: ...
def get_cuda_arch_flags(cflags: Any): ...
def get_rocm_arch_flags(cflags: Any): ...
def normalize_extension_kwargs(kwargs: Any, use_cuda: bool = ...): ...
def create_sym_link_if_not_exist(): ...
def find_cuda_home(): ...
def find_rocm_home(): ...
def find_cuda_includes(): ...
def find_rocm_includes(): ...
def find_paddle_includes(use_cuda: bool = ...): ...
def find_clang_cpp_include(compiler: str = ...): ...
def find_cuda_libraries(): ...
def find_rocm_libraries(): ...
def find_paddle_libraries(use_cuda: bool = ...): ...
def add_compile_flag(extra_compile_args: Any, flags: Any) -> None: ...
def is_cuda_file(path: Any): ...
def get_build_directory(verbose: bool = ...): ...
def parse_op_info(op_name: Any): ...
def list2str(args: Any): ...
def parse_op_name_from(sources: Any): ...
def run_cmd(command: Any, verbose: bool = ...): ...
def check_abi_compatibility(compiler: Any, verbose: bool = ...): ...
def log_v(info: Any, verbose: bool = ...) -> None: ...
