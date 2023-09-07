from __future__ import annotations

from distutils.command.build import build
from typing import Any, Optional

from setuptools.command.build_ext import build_ext
from setuptools.command.easy_install import easy_install

from ...base import core as core
from .extension_utils import CLANG_COMPILE_FLAGS as CLANG_COMPILE_FLAGS
from .extension_utils import CLANG_LINK_FLAGS as CLANG_LINK_FLAGS
from .extension_utils import IS_WINDOWS as IS_WINDOWS
from .extension_utils import MSVC_COMPILE_FLAGS as MSVC_COMPILE_FLAGS
from .extension_utils import OS_NAME as OS_NAME
from .extension_utils import CustomOpInfo as CustomOpInfo
from .extension_utils import add_compile_flag as add_compile_flag
from .extension_utils import add_std_without_repeat as add_std_without_repeat
from .extension_utils import bootstrap_context as bootstrap_context
from .extension_utils import check_abi_compatibility as check_abi_compatibility
from .extension_utils import (
    clean_object_if_change_cflags as clean_object_if_change_cflags,
)
from .extension_utils import find_cuda_home as find_cuda_home
from .extension_utils import find_rocm_home as find_rocm_home
from .extension_utils import get_build_directory as get_build_directory
from .extension_utils import is_cuda_file as is_cuda_file
from .extension_utils import log_v as log_v
from .extension_utils import normalize_extension_kwargs as normalize_extension_kwargs
from .extension_utils import parse_op_name_from as parse_op_name_from
from .extension_utils import prepare_unix_cudaflags as prepare_unix_cudaflags
from .extension_utils import prepare_win_cudaflags as prepare_win_cudaflags
from .extension_utils import run_cmd as run_cmd

CUDA_HOME: Any
ROCM_HOME: Any
CUDA_HOME = ROCM_HOME

def setup(**attr: Any) -> None: ...
def CppExtension(sources: Any, *args: Any, **kwargs: Any): ...
def CUDAExtension(sources: Any, *args: Any, **kwargs: Any): ...

class BuildExtension(build_ext):
    @classmethod
    def with_options(cls, **options: Any): ...
    no_python_abi_suffix: Any = ...
    output_dir: Any = ...
    contain_cuda_file: bool = ...
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def initialize_options(self) -> None: ...
    build_lib: Any = ...
    def finalize_options(self) -> None: ...
    cflags: Any = ...
    def build_extensions(self): ...
    def get_ext_filename(self, fullname: Any): ...

class EasyInstallCommand(easy_install):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def run(self, *args: Any, **kwargs: Any) -> None: ...

class BuildCommand(build):
    @classmethod
    def with_options(cls, **options: Any): ...
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    build_base: Any = ...
    def initialize_options(self) -> None: ...

def load(
    name: Any,
    sources: Any,
    extra_cxx_cflags: Any | None = ...,
    extra_cuda_cflags: Any | None = ...,
    extra_ldflags: Any | None = ...,
    extra_include_paths: Any | None = ...,
    build_directory: Any | None = ...,
    verbose: bool = ...,
): ...
