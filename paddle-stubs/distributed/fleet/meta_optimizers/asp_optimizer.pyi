from __future__ import annotations

from typing import Any, Optional

from .meta_optimizer_base import MetaOptimizerBase as MetaOptimizerBase

class ASPOptimizer(MetaOptimizerBase):
    inner_opt: Any = ...
    meta_optimizers_white_list: Any = ...
    meta_optimizers_black_list: Any = ...
    def __init__(self, optimizer: Any) -> None: ...
    def minimize_impl(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
    ): ...
