from __future__ import annotations

from typing import Any, Optional

from .base_cost import OP_COST_FACTORY as OP_COST_FACTORY
from .base_cost import CommOpCost as CommOpCost
from .base_cost import register_op_cost as register_op_cost

class AllreduceSumCost(CommOpCost):
    OP_TYPE: str = ...
    def __init__(
        self, op: Any | None = ..., op_desc: Any | None = ..., comm_context: Any | None = ...
    ) -> None: ...
    def calc_time(self): ...
