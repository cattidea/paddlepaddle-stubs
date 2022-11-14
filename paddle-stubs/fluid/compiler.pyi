from __future__ import annotations

from typing import Any, Optional

ExecutionStrategy: Any
BuildStrategy: Any

class CompiledProgram:
    def __init__(self, program_or_graph: Any, build_strategy: Any | None = ...) -> None: ...
    def with_data_parallel(
        self,
        loss_name: str | None = ...,
        build_strategy: Any | None = ...,
        exec_strategy: Any | None = ...,
        share_vars_from: Any | None = ...,
        places: Any | None = ...,
    ): ...

class IpuStrategy:
    has_custom_ops: bool = ...
    custom_op_names: Any = ...
    def __init__(self) -> None: ...
    def set_graph_config(
        self, num_ipus: int = ..., is_training: bool = ..., micro_batch_size: int = ..., enable_manual_shard: bool = ...
    ) -> None: ...
    def set_pipelining_config(
        self,
        enable_pipelining: bool = ...,
        batches_per_step: int = ...,
        enable_gradient_accumulation: bool = ...,
        accumulation_factor: int = ...,
    ) -> None: ...
    def set_precision_config(self, enable_fp16: bool = ...) -> None: ...
    def add_custom_op(
        self, paddle_op: Any, popart_op: Any | None = ..., domain: str = ..., version: int = ...
    ) -> None: ...
    def set_options(self, options: Any) -> None: ...
    def get_option(self, option: Any): ...
    def enable_pattern(self, pattern: Any) -> None: ...
    def disable_pattern(self, pattern: Any) -> None: ...
    @property
    def num_ipus(self): ...
    @property
    def is_training(self): ...
    @property
    def enable_pipelining(self): ...
    @property
    def enable_fp16(self): ...

class IpuCompiledProgram:
    def __init__(
        self, program: Any | None = ..., scope: Any | None = ..., ipu_strategy: Any | None = ...
    ) -> None: ...
    def compile(self, feed_list: Any, fetch_list: Any): ...
