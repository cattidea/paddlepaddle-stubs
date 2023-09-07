from __future__ import annotations

from typing import Any

from paddle.base import core as core
from paddle.base import framework as framework
from paddle.base.framework import Variable as Variable
from paddle.optimizer import Optimizer as Optimizer

from ...base.topology import ParallelMode as ParallelMode

class HybridParallelGradScaler:
    def __init__(self, scaler: Any, hcg: Any) -> None: ...
    def scale(self, var: Any): ...
    def minimize(self, optimizer: Any, *args: Any, **kwargs: Any): ...
    def __getattr__(self, item: Any): ...
