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
from ..fluid import core as core
from ..fluid.core import VarBase as VarBase
from ..fluid.dygraph.base import grad as grad
from ..fluid.dygraph.parallel import DataParallel as DataParallel
from ..fluid.framework import OpProtoHolder as OpProtoHolder
from ..fluid.framework import convert_np_dtype_to_dtype_ as convert_np_dtype_to_dtype_
from ..fluid.framework import disable_signal_handler as disable_signal_handler
from ..fluid.framework import dygraph_only as dygraph_only
from ..fluid.framework import get_flags as get_flags
from ..fluid.framework import monkey_patch_math_varbase as monkey_patch_math_varbase
from ..fluid.framework import set_flags as set_flags
from ..fluid.layers import monkey_patch_variable as monkey_patch_variable
from ..fluid.layers.tensor import create_parameter as create_parameter
from ..fluid.param_attr import ParamAttr as ParamAttr
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
