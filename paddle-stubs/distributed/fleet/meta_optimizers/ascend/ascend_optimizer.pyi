from __future__ import annotations

from collections import namedtuple
from typing import Any, Optional

from paddle.fluid.optimizer import Optimizer

from . import ascend_parser as ascend_parser

HcomGroupConfig = namedtuple("HcomGroupConfig", ["name", "nranks", "rank_ids"])

class AscendIRParser:
    graph_idx: int = ...
    hcom_endpoints: Any = ...
    groups_to_create: Any = ...
    def __init__(self, auto_dp: bool = ..., world_rank_size: int = ...) -> None: ...
    def parse_op(self, op: Any) -> None: ...
    def parse_program(self, startup_program: Any, main_program: Any, input_varlist: Any, fetch_list: Any): ...

class AscendOptimizer(Optimizer):
    inner_opt: Any = ...
    fetch_list: Any = ...
    ascend_instance: Any = ...
    def __init__(self, optimizer: Any, fetch_list: Any = ...) -> None: ...
    def __del__(self) -> None: ...
    parser: Any = ...
    def minimize(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
        auto_dp: bool = ...,
        rank_table_file: Any | None = ...,
        precision_mode: str = ...,
    ): ...
