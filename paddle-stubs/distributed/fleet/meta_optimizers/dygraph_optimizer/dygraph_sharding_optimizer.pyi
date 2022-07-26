from __future__ import annotations

from typing import Any, Optional

from ...utils.log_util import logger as logger

class DygraphShardingOptimizer:
    def __init__(
        self,
        hcg: Any,
        user_defined_strategy: Any,
        params: Any,
        inner_optimizer_class: Any,
        **inner_optimizer_kargs: Any,
    ) -> None: ...
    def clear_grad(self) -> None: ...
    def minimize(
        self,
        loss: Any,
        startup_program: Optional[Any] = ...,
        parameters: Optional[Any] = ...,
        no_grad_set: Optional[Any] = ...,
    ): ...
    def step(self) -> None: ...
    def __getattr__(self, item: Any): ...
