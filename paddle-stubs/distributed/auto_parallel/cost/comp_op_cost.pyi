from __future__ import annotations

from typing import Any, Optional

from .base_cost import OP_COST_FACTORY as OP_COST_FACTORY
from .base_cost import CompOpCost as CompOpCost
from .base_cost import Cost as Cost
from .base_cost import register_op_cost as register_op_cost

class MatmulV2OpCost(CompOpCost):
    OP_TYPE: str = ...
    def __init__(self, op: Any | None = ..., op_desc: Any | None = ..., cluster: Any | None = ...) -> None: ...
    def calc_flops(self): ...
    def calc_time(self): ...
