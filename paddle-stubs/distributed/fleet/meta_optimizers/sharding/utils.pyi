from __future__ import annotations

from typing import Any, Optional

from paddle.distributed.fleet.meta_optimizers.common import (
    OP_ROLE_VAR_KEY as OP_ROLE_VAR_KEY,
)

def check_broadcast(block: Any) -> None: ...
def check_allreduce_sum(block: Any, shard: Any, sharding_ring_id: Any, dp_ring_id: int = ...) -> None: ...
def get_valid_op_role(block: Any, insert_idx: Any): ...
def insert_sync_calc_op(block: Any, insert_idx: Any, calc_dep_vars: Any) -> None: ...
def insert_sync_comm_op(block: Any, insert_idx: Any, ring_id: Any, comm_dep_vars: Any): ...
def insert_sync_comm_ops(block: Any, insert_idx: Any, ring_id: Any, comm_dep_vars: Any): ...
def insert_fill_constant_ops(block: Any, insert_idx: Any, fill_constant_vars: Any) -> None: ...
def insert_cast_ops(block: Any, insert_idx: Any, cast_ops: Any) -> None: ...
def insert_allreduce_ops(
    block: Any,
    insert_idx: Any,
    ring_id: Any,
    allreduce_vars: Any,
    op_role: Any = ...,
    use_calc_stream: bool = ...,
    user_defined_strategy: Optional[Any] = ...,
) -> None: ...

class FuseHelper:
    @staticmethod
    def sort_vars_by_dtype(block: Any, vars_name: Any): ...
    @staticmethod
    def get_fused_groups(block: Any, vars_name: Any, fuse_size: float = ...): ...
    @staticmethod
    def insert_coalesce_tensor(block: Any, index: Any, groups: Any, op_role: Any = ..., prefix: str = ...): ...

def insert_fused_allreduce_ops(
    block: Any,
    insert_idx: Any,
    ring_id: Any,
    allreduce_vars: Any,
    op_role: Any = ...,
    use_calc_stream: bool = ...,
    fuse_grad_size_in_MB: int = ...,
) -> None: ...
def insert_fused_reduce_ops(
    block: Any,
    insert_idx: Any,
    ring_id: Any,
    reduce_vars: Any,
    shard: Any,
    op_role: Any = ...,
    use_calc_stream: bool = ...,
    rank: Optional[Any] = ...,
    fuse_grad_size: int = ...,
): ...
def insert_reduce_ops(
    block: Any,
    insert_idx: Any,
    ring_id: Any,
    reduce_vars: Any,
    shard: Any,
    op_role: Any = ...,
    use_calc_stream: bool = ...,
    rank: Optional[Any] = ...,
    strategy: Optional[Any] = ...,
): ...
def insert_fused_broadcast_param_ops(
    block: Any,
    insert_idx: Any,
    ring_id: Any,
    params: Any,
    shard: Any,
    op_role: Any = ...,
    use_calc_stream: bool = ...,
    rank: Optional[Any] = ...,
    fuse_size: int = ...,
): ...
def insert_broadcast_param_ops(
    block: Any,
    insert_idx: Any,
    ring_id: Any,
    params: Any,
    shard: Any,
    op_role: Any = ...,
    use_calc_stream: bool = ...,
    rank: Optional[Any] = ...,
    strategy: Optional[Any] = ...,
): ...
def fuse_opt_broadcast_param_ops(
    block: Any, ring_id: Any, shard: Any, op_role: Any = ..., strategy: Optional[Any] = ...
) -> None: ...
def get_grad_device(grad_name: Any, shard: Any): ...
def get_first_check_finite_and_unscale_op_idx(block: Any, raise_error: bool = ...): ...
def get_first_optimize_op_idx(block: Any): ...
def insert_broadcast_ops(block: Any, insert_idx: Any, ring_id: Any, broadcast2root: Any) -> None: ...

DtypeToSize: Any

def get_var_size(param: Any): ...
def insert_scale_loss_grad_ops(block: Any, scale: float = ...) -> None: ...
def comm_analyse(main_program: Any): ...
def add_sync_comm(program: Any, sharding_ring_id: Any) -> None: ...
def save_persistables(exe: Any, dirname: Any, main_program: Any, filename: Optional[str] = ...): ...
def append_naive_sync(block: Any, sync_var: Any, ring_id: Any) -> None: ...
