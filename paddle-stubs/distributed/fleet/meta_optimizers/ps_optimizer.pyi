from __future__ import annotations

from typing import Any, Optional

from paddle.distributed.ps.utils.public import *

from ..base.private_helper_function import wait_server_ready as wait_server_ready
from .meta_optimizer_base import MetaOptimizerBase as MetaOptimizerBase

class ParameterServerOptimizer(MetaOptimizerBase):
    inner_opt: Any = ...
    meta_optimizers_white_list: Any = ...
    def __init__(self, optimizer: Any) -> None: ...
    def minimize_impl(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
    ): ...
    def minimize_losses_impl(
        self,
        losses: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
    ): ...
