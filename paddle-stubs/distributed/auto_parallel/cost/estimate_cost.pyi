from __future__ import annotations

from typing import Any, Optional

class CostEstimator:
    def __init__(
        self, program: Any, cluster: Any | None = ..., dist_context: Any | None = ..., mode: str = ...
    ) -> None: ...
    @property
    def program(self): ...
    @property
    def dist_context(self): ...
    @property
    def cluster(self): ...
    @property
    def mode(self): ...
    @property
    def global_cost(self): ...
    @property
    def local_cost(self): ...
    def get_op_cost(self): ...
    def get_tensor_cost(self): ...
    def get_global_cost(self): ...
    def get_local_cost(self, rank: Any | None = ...): ...
