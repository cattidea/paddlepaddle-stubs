from paddle.base.libpaddle.pir import (
    Block,
    Operation,
    OpOperand,
    OpResult,
    PassManager,
    Program,
    Type,
    Value,
    check_unregistered_ops,
    fake_op_result,
    is_fake_op_result,
    register_paddle_dialect,
    reset_insertion_point_to_end,
    reset_insertion_point_to_start,
    set_global_program,
    set_insertion_point,
    translate_to_new_ir,
    translate_to_new_ir_with_param_map,
)

from . import core
from .math_op_patch import monkey_patch_opresult
