from __future__ import annotations

from enum import Enum
from typing import Any, Optional

SUCC: int
PRED: int

class CostNodeType(Enum):
    DEFAULT = ...
    COMPUTATION = ...
    COMMUNICATION = ...
    VARIABLE = ...
    MERGED = ...
    NOP = ...

class Cost:
    runtime: Any = ...
    static_mem: Any = ...
    peak_mem: Any = ...
    def __init__(self) -> None: ...

class CostModelMode(Enum):
    DEFAULT = ...
    BENCHMARKING = ...
    ANALYSIS = ...
    MIXED = ...

class CostNode:
    id: Any = ...
    node: Any = ...
    type: Any = ...
    is_optim: bool = ...
    is_bwd: bool = ...
    def __init__(self, node: Any, node_type: Any, id: Any | None = ...) -> None: ...
    @property
    def cost(self): ...
    @cost.setter
    def cost(self, cost: Any) -> None: ...

class MergedOpsCostNode(CostNode):
    node_list: Any = ...
    is_bwd: Any = ...
    def __init__(
        self, node_type: Any, id: Any | None = ..., base_node_list: Any | None = ..., is_bwd: bool = ...
    ) -> None: ...

class CommOpCostNode(CostNode):
    node_list: Any = ...
    ranks: Any = ...
    comm_type: Any = ...
    is_bwd: Any = ...
    def __init__(
        self,
        node: Any,
        node_type: Any,
        id: Any | None = ...,
        comm_node_list: Any | None = ...,
        is_bwd: bool = ...,
    ) -> None: ...
    def set_ranks(self, ranks: Any) -> None: ...
    input_shape: Any = ...
    output_shape: Any = ...
    def set_shapes(self, input_shape: Any, output_shape: Any) -> None: ...
    def init_comm_cost(self, cluster: Any | None = ...) -> None: ...

class TensorCostNode(CostNode):
    shape: Any = ...
    dtype: Any = ...
    dtype_factor: int = ...
    persistable: Any = ...
    shared_node_id: Any = ...
    batch_size: Any = ...
    def __init__(
        self,
        node: Any,
        node_type: Any,
        id: Any | None = ...,
        base_node_list: Any | None = ...,
        batch_size: Any | None = ...,
        shared_node_id: Any | None = ...,
    ) -> None: ...
    def get_size(self): ...

class CompOpCostNode(CostNode):
    is_bwd: Any = ...
    is_optim: Any = ...
    def __init__(
        self, node: Any, node_type: Any, id: Any | None = ..., is_bwd: bool = ..., is_optim: bool = ...
    ) -> None: ...
    cost: Any = ...
    def init_comp_cost(self, cost_data: Any) -> None: ...

class PipeEvent:
    stage_id: Any = ...
    name: Any = ...
    duration: Any = ...
    s_time: Any = ...
    e_time: int = ...
    def __init__(self, stage_id: Any, event_name: Any, duration: Any, start_time: int = ...) -> None: ...

class CostModel:
    mode: Any = ...
    opcall_overhead: Any = ...
    batch_size: Any = ...
    microbatch_num: Any = ...
    nodes: Any = ...
    origin_graph: Any = ...
    op_graph: Any = ...
    runtime_graph: Any = ...
    cluster: Any = ...
    cost_data: Any = ...
    pp2rank: Any = ...
    rank2pp: Any = ...
    ring2rank: Any = ...
    fwd_time: Any = ...
    bwd_time: Any = ...
    optim_time: Any = ...
    def __init__(
        self,
        mode: Any = ...,
        cluster: Any | None = ...,
        batch_size: int = ...,
        microbatch_num: int = ...,
        opcall_overhead: int = ...,
        standalone_cost_data: Any | None = ...,
        pipeline_config: Any | None = ...,
    ) -> None: ...
    distributed_program: Any = ...
    total_rank: Any = ...
    def parse_program(self, distributed_program: Any): ...
    def build_op_graph(self) -> None: ...
    def build_runtime_graph(self) -> None: ...
    def eliminate_multi_edges(self, graph: Any | None = ...) -> None: ...
    def merge_comm(self) -> None: ...
    def merge_linear(self): ...
    def merge_branch(self): ...
    def get_runtime_cost(self): ...
    def get_mem(self): ...
    def get_pipeline_time(self): ...
    def get_cost(self): ...
    def init(self, distributed_program: Any) -> None: ...

def estimate_cost(
    distributed_program: Any, cluster: Any, pipeline_config: Any, standalone_cost_data: Any, batch_size: Any
): ...
