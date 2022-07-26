from __future__ import annotations

from typing import Any

from paddle.fluid.incubate.fleet.parameter_server.ir.trainer_pass import (
    find_op_input_output as find_op_input_output,
)
from paddle.fluid.incubate.fleet.parameter_server.ir.trainer_pass import (
    get_vars_name_in_block as get_vars_name_in_block,
)
from paddle.fluid.transpiler.details.program_utils import delete_ops as delete_ops

def split_heter_worker_ops_pass(program: Any, config: Any, stage_id: Any, device: Any): ...
def split_trainer_ops_pass(program: Any, config: Any, default_device: str = ...): ...
