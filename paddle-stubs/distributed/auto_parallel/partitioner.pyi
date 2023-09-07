from __future__ import annotations

from typing import Any

from paddle.base import unique_name as unique_name
from paddle.base.framework import Variable as Variable
from paddle.base.framework import program_guard as program_guard
from paddle.distributed.auto_parallel.dist_context import (
    DistributedOperatorContext as DistributedOperatorContext,
)

from .dist_attribute import OperatorDistributedAttribute as OperatorDistributedAttribute
from .operators.common import BACKWARD_ONLY_DIST_OPS as BACKWARD_ONLY_DIST_OPS
from .process_group import new_process_group as new_process_group
from .utils import is_backward_op as is_backward_op
from .utils import is_forward_op as is_forward_op
from .utils import print_program_with_dist_attr as print_program_with_dist_attr
from .utils import set_dist_op_desc_original_id as set_dist_op_desc_original_id

__varname_not_in_block__: Any
__not_shape_var_type__: Any

class Partitioner:
    def __init__(self, dist_context: Any, rank_id: int = ...) -> None: ...
    def partition(self, serial_main_program: Any, serial_startup_program: Any, params_grads: Any): ...
    def partition_startup_program(self, serial_main_program: Any, serial_startup_program: Any): ...
    def partition_main_program(self, serial_main_program: Any, params_and_grads: Any): ...
    def partition_block(self, ref_block: Any, target_block: Any) -> None: ...
