from __future__ import annotations

from typing import Any, Optional

from .... import core as core
from .... import unique_name as unique_name
from ....executor import global_scope as global_scope
from ....framework import Operator as Operator
from ....framework import Program as Program
from ....framework import Variable as Variable
from ....framework import program_guard as program_guard
from ....initializer import Constant as Constant
from ....layer_helper import LayerHelper as LayerHelper
from ....log_helper import get_logger as get_logger
from ....param_attr import ParamAttr as ParamAttr

def find_next_ops(block: Any, var_name: Any): ...
def load_variable_data(scope: Any, var_name: Any): ...

class QuantizeTranspilerV2:
    def __init__(
        self,
        weight_bits: int = ...,
        activation_bits: int = ...,
        weight_quantize_type: str = ...,
        activation_quantize_type: str = ...,
        quantizable_op_type: Any = ...,
        skip_pattern: Any = ...,
    ) -> None: ...
    def apply(self, program: Any, startup_program: Any, is_test: bool = ...) -> None: ...
    def convert(self, test_program: Any, scope: Optional[Any] = ...) -> None: ...
