from __future__ import annotations

from typing import Any

from paddle.base.data_feeder import convert_dtype as convert_dtype

from .. import core as core
from .. import framework as framework
from .. import unique_name as unique_name
from ..framework import EagerParamBase as EagerParamBase
from ..framework import ParamBase as ParamBase
from ..framework import Parameter as Parameter
from ..framework import Variable as Variable
from ..framework import convert_np_dtype_to_dtype_ as convert_np_dtype_to_dtype_
from ..framework import in_dygraph_mode as in_dygraph_mode
from .math_op_patch import monkey_patch_math_varbase as monkey_patch_math_varbase

class TensorHookRemoveHelper:
    def __init__(self, tensor: Any, hook_id: Any) -> None: ...
    def remove(self): ...

def monkey_patch_varbase(): ...
