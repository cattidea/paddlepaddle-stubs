from __future__ import annotations

from .. import core as core
from .. import framework as framework
from ..framework import Variable as Variable
from ..framework import convert_np_dtype_to_dtype_ as convert_np_dtype_to_dtype_
from ..framework import in_dygraph_mode as in_dygraph_mode
from ..layers.layer_function_generator import OpProtoHolder as OpProtoHolder
from . import no_grad as no_grad

def monkey_patch_math_varbase(): ...
