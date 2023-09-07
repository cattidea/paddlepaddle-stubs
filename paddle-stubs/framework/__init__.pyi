from __future__ import annotations

from typing import Callable

from typing_extensions import ParamSpec, TypeVar

_InputArgs = ParamSpec("_InputArgs")
_RetValue = TypeVar("_RetValue")

from .._typing import CPUPlace as CPUPlace
from .._typing import CUDAPinnedPlace as CUDAPinnedPlace
from .._typing import CUDAPlace as CUDAPlace
from .._typing import CustomPlace as CustomPlace
from .._typing import IPUPlace as IPUPlace
from .._typing import MLUPlace as MLUPlace
from .._typing import NPUPlace as NPUPlace
from ..base import core as core
from ..base.core import VarBase as VarBase
from ..base.dygraph.base import grad as grad
from ..base.dygraph.parallel import DataParallel as DataParallel
from ..base.framework import OpProtoHolder as OpProtoHolder
from ..base.framework import convert_np_dtype_to_dtype_ as convert_np_dtype_to_dtype_
from ..base.framework import disable_signal_handler as disable_signal_handler
from ..base.framework import dygraph_only as dygraph_only
from ..base.framework import get_flags as get_flags
from ..base.framework import monkey_patch_math_varbase as monkey_patch_math_varbase
from ..base.framework import set_flags as set_flags
from ..base.layers import monkey_patch_variable as monkey_patch_variable
from ..base.layers.tensor import create_parameter as create_parameter
from ..base.param_attr import ParamAttr as ParamAttr
from . import random as random
from .framework import get_default_dtype as get_default_dtype
from .framework import is_grad_enabled as is_grad_enabled
from .framework import set_default_dtype as set_default_dtype
from .framework import set_grad_enabled as set_grad_enabled
from .io import load as load
from .io import save as save
from .random import seed as seed

def no_grad(func: Callable[_InputArgs, _RetValue]) -> Callable[_InputArgs, _RetValue]: ...
def in_dynamic_mode() -> bool: ...
def enable_static() -> None: ...
def disable_static() -> None: ...
