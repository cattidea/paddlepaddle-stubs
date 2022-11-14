from __future__ import annotations

from typing import Any, Optional

class HeterParallelOptimizer:
    def __init__(self, optimizer: Any, strategy: Any) -> None: ...
    def step(self) -> None: ...
    def minimize(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameters: Any | None = ...,
        no_grad_set: Any | None = ...,
    ): ...
    def __getattr__(self, item: Any): ...
