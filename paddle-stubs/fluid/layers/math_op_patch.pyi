from __future__ import annotations

from typing import Any

from .. import core as core
from ..framework import Variable as Variable
from ..framework import static_only as static_only
from ..framework import unique_name as unique_name
from .control_flow import array_length as array_length
from .control_flow import array_write as array_write
from .layer_function_generator import OpProtoHolder as OpProtoHolder

compare_ops: Any
EXPRESSION_MAP: Any

def monkey_patch_variable(): ...
