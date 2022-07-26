from __future__ import annotations

from collections import OrderedDict as OrderedDict
from typing import Any

from paddle.fluid import unique_name as unique_name
from paddle.fluid.clip import append_gradient_clip_ops as append_gradient_clip_ops
from paddle.fluid.framework import program_guard as program_guard

from .pass_base import PassBase as PassBase
from .pass_base import PassType as PassType
from .pass_base import register_pass as register_pass

world_process_group: Any

def parse_program(
    main_program: Any, startup_program: Any, params_grads: Any, k_steps: Any, avg: Any, dist_context: Any
) -> None: ...

class GradientMergePass(PassBase):
    def __init__(self) -> None: ...
