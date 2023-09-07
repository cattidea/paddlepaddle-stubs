from __future__ import annotations

from typing import Any, Optional

import paddle.base as base
from paddle.base.compiler import CompiledProgram as CompiledProgram
from paddle.base.incubate.fleet.base.fleet_base import DistributedOptimizer, Fleet
from paddle.base.parallel_executor import ParallelExecutor as ParallelExecutor

class LambConfig:
    def __init__(self) -> None: ...

class DistFCConfig:
    def __init__(self) -> None: ...

class Collective(Fleet):
    startup_program: Any = ...
    main_program: Any = ...
    def __init__(self) -> None: ...
    def init_worker(self) -> None: ...
    def run_worker(self, main_programs: Any | None = ..., scopes: Any | None = ...) -> None: ...
    def init_server(self, model_dir: Any | None = ...) -> None: ...
    def run_server(self) -> None: ...
    def stop_worker(self) -> None: ...
    def distributed_optimizer(self, optimizer: Any, strategy: Any | None = ...): ...
    def save_inference_model(
        self,
        executor: Any,
        dirname: Any,
        feeded_var_names: Any | None = ...,
        target_vars: Any | None = ...,
        main_program: Any | None = ...,
        export_for_deployment: bool = ...,
    ) -> None: ...
    def save_persistables(
        self, executor: Any, dirname: Any, main_program: Any | None = ..., filename: str | None = ...
    ) -> None: ...
    def save_checkpoint(
        self,
        executor: Any,
        path: Any,
        trainer_id: Any,
        train_status: Any,
        fs: Any,
        main_program: Any | None = ...,
        local_cache_path: str = ...,
        remain_all_checkpoint: bool = ...,
    ): ...
    def load_checkpoint(
        self,
        executor: Any,
        path: Any,
        trainer_id: Any,
        train_status: Any,
        fs: Any,
        main_program: Any | None = ...,
        local_cache_path: str = ...,
        ignore_empty: bool = ...,
    ): ...

fleet: Any

class DistributedStrategy(base.BuildStrategy):
    use_local_sgd: bool = ...
    use_dist_fc: bool = ...
    dist_fc_config: Any = ...
    mode: str = ...
    collective_mode: Any = ...
    nccl_comm_num: int = ...
    forward_recompute: bool = ...
    recompute_checkpoints: Any = ...
    use_amp: bool = ...
    amp_loss_scaling: Any = ...
    exec_strategy: Any = ...
    def __init__(self) -> None: ...

class CollectiveOpBasedOptimizer(DistributedOptimizer):
    def __init__(self, optimizer: Any, strategy: Any | None = ...) -> None: ...
    def backward(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
        callbacks: Any | None = ...,
    ): ...
    def apply_gradients(self, params_grads: Any): ...

class CollectiveOptimizer(DistributedOptimizer):
    print_config: bool = ...
    def __init__(self, optimizer: Any, strategy: Any = ...) -> None: ...
    def backward(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
        callbacks: Any | None = ...,
    ): ...
    def apply_gradients(self, params_grads: Any): ...
    def raiseOptimizeError(self, strategy_name: Any, optimize_name: Any) -> None: ...
    def minimize(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
    ): ...