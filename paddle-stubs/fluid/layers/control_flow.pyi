from __future__ import annotations

from typing import Any, Optional

def Print(
    input: Any,
    first_n: int = ...,
    message: Any | None = ...,
    summarize: int = ...,
    print_tensor_name: bool = ...,
    print_tensor_type: bool = ...,
    print_tensor_shape: bool = ...,
    print_tensor_layout: bool = ...,
    print_tensor_lod: bool = ...,
    print_phase: str = ...,
): ...
def Assert(cond: Any, data: Any | None = ..., summarize: int = ..., name: str | None = ...): ...

class BlockGuard:
    main_program: Any = ...
    def __init__(self, main_program: Any) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any): ...

class BlockGuardWithCompletion(BlockGuard):
    rnn: Any = ...
    def __init__(self, rnn: Any) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any): ...

class StaticRNNMemoryLink:
    init: Any = ...
    pre_mem: Any = ...
    mem: Any = ...
    def __init__(self, init: Any, pre_mem: Any, mem: Any | None = ...) -> None: ...

class StaticRNN:
    BEFORE_RNN_BLOCK: int = ...
    IN_RNN_BLOCK: int = ...
    AFTER_RNN_BLOCK: int = ...
    helper: Any = ...
    memories: Any = ...
    inputs: Any = ...
    outputs: Any = ...
    status: Any = ...
    seq_len: Any = ...
    def __init__(self, name: str | None = ...) -> None: ...
    def step(self): ...
    def memory(
        self,
        init: Any | None = ...,
        shape: Any | None = ...,
        batch_ref: Any | None = ...,
        init_value: float = ...,
        init_batch_dim_idx: int = ...,
        ref_batch_dim_idx: int = ...,
    ): ...
    def step_input(self, x: Any): ...
    def step_output(self, o: Any) -> None: ...
    def output(self, *outputs: Any) -> None: ...
    def update_memory(self, mem: Any, var: Any) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any): ...

class WhileGuard(BlockGuard):
    while_op: Any = ...
    def __init__(self, while_op: Any) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any): ...

class While:
    BEFORE_WHILE_BLOCK: int = ...
    IN_WHILE_BLOCK: int = ...
    AFTER_WHILE_BLOCK: int = ...
    helper: Any = ...
    status: Any = ...
    cond_var: Any = ...
    is_test: Any = ...
    def __init__(self, cond: Any, is_test: bool = ..., name: str | None = ...): ...
    def block(self): ...

def while_loop(cond: Any, body: Any, loop_vars: Any, is_test: bool = ..., name: str | None = ...): ...
def increment(x: Any, value: float = ..., in_place: bool = ...): ...
def array_write(x: Any, i: Any, array: Any | None = ...): ...
def create_array(dtype: Any, initialized_list: Any | None = ...): ...
def less_than(x: Any, y: Any, force_cpu: Any | None = ..., cond: Any | None = ..., name: str | None = ...): ...
def less_equal(x: Any, y: Any, cond: Any | None = ..., name: str | None = ...): ...
def greater_than(x: Any, y: Any, cond: Any | None = ..., name: str | None = ...): ...
def greater_equal(x: Any, y: Any, cond: Any | None = ..., name: str | None = ...): ...
def equal(x: Any, y: Any, cond: Any | None = ..., name: str | None = ...): ...
def not_equal(x: Any, y: Any, cond: Any | None = ..., name: str | None = ...): ...
def array_read(array: Any, i: Any): ...
def array_length(array: Any): ...

class ConditionalBlockGuard(BlockGuard):
    block: Any = ...
    def __init__(self, block: Any) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any): ...

class ConditionalBlock:
    inputs: Any = ...
    is_scalar_condition: Any = ...
    helper: Any = ...
    def __init__(self, inputs: Any, is_scalar_condition: bool = ..., name: str | None = ...) -> None: ...
    def block(self): ...
    def complete(self) -> None: ...
    def need_append_conditional_block_grad(self, inside_block: Any): ...
    def append_conditional_block_grad(
        self, parent_block: Any, inside_block: Any, conditional_block_op: Any
    ) -> None: ...

def cond(pred: Any, true_fn: Any | None = ..., false_fn: Any | None = ..., name: str | None = ...): ...
def case(pred_fn_pairs: Any, default: Any | None = ..., name: str | None = ...): ...

class Switch:
    helper: Any = ...
    inside_scope: bool = ...
    pre_not_conditions: Any = ...
    def __init__(self, name: str | None = ...) -> None: ...
    def case(self, condition: Any): ...
    def default(self): ...
    def __enter__(self): ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any): ...

class IfElseBlockGuard:
    is_true: Any = ...
    ie: Any = ...
    cond_block: Any = ...
    def __init__(self, is_true: Any, ifelse: Any) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any): ...

class IfElse:
    OUT_IF_ELSE_BLOCKS: int = ...
    IN_IF_ELSE_TRUE_BLOCKS: int = ...
    IN_IF_ELSE_FALSE_BLOCKS: int = ...
    helper: Any = ...
    cond: Any = ...
    input_table: Any = ...
    status: Any = ...
    conditional_true_block: Any = ...
    conditional_false_block: Any = ...
    output_table: Any = ...
    def __init__(self, cond: Any, name: str | None = ...) -> None: ...
    def input(self, x: Any): ...
    def true_block(self): ...
    def false_block(self): ...
    def output(self, *outs: Any) -> None: ...
    def __call__(self): ...

class DynamicRNN:
    BEFORE_RNN: int = ...
    IN_RNN: int = ...
    AFTER_RNN: int = ...
    helper: Any = ...
    status: Any = ...
    lod_rank_table: Any = ...
    max_seq_len: Any = ...
    step_idx: Any = ...
    zero_idx: Any = ...
    mem_dict: Any = ...
    output_array: Any = ...
    outputs: Any = ...
    cond: Any = ...
    while_op: Any = ...
    input_array: Any = ...
    mem_link: Any = ...
    def __init__(self, name: str | None = ...) -> None: ...
    def step_input(self, x: Any, level: int = ...): ...
    def static_input(self, x: Any): ...
    def block(self) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any): ...
    def memory(
        self,
        init: Any | None = ...,
        shape: Any | None = ...,
        value: float = ...,
        need_reorder: bool = ...,
        dtype: str = ...,
    ): ...
    def update_memory(self, ex_mem: Any, new_mem: Any) -> None: ...
    def output(self, *outputs: Any) -> None: ...

def switch_case(branch_index: Any, branch_fns: Any, default: Any | None = ..., name: str | None = ...): ...
def reorder_lod_tensor_by_rank(x: Any, rank_table: Any): ...
def is_empty(x: Any, name: str | None = ...): ...
