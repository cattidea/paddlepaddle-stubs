from __future__ import annotations

from typing import Any, Optional

import paddle.fluid as fluid
from paddle.fluid.compiler import CompiledProgram as CompiledProgram
from paddle.fluid.incubate.fleet.base.fleet_base import DistributedOptimizer, Fleet
from paddle.fluid.parallel_executor import ParallelExecutor as ParallelExecutor

class LambConfig:
    def __init__(self) -> None: ...

class DistFCConfig:
    def __init__(self) -> None: ...

class Collective(Fleet):
    startup_program: Any = ...
    main_program: Any = ...
    def __init__(self) -> None: ...
    def init_worker(self) -> None: ...
    def run_worker(self, main_programs: Optional[Any] = ..., scopes: Optional[Any] = ...) -> None: ...
    def init_server(self, model_dir: Optional[Any] = ...) -> None: ...
    def run_server(self) -> None: ...
    def stop_worker(self) -> None: ...
    def distributed_optimizer(self, optimizer: Any, strategy: Optional[Any] = ...): ...
    def save_inference_model(
        self,
        executor: Any,
        dirname: Any,
        feeded_var_names: Optional[Any] = ...,
        target_vars: Optional[Any] = ...,
        main_program: Optional[Any] = ...,
        export_for_deployment: bool = ...,
    ) -> None: ...
    def save_persistables(
        self, executor: Any, dirname: Any, main_program: Optional[Any] = ..., filename: Optional[Any] = ...
    ) -> None: ...
    def save_checkpoint(
        self,
        executor: Any,
        path: Any,
        trainer_id: Any,
        train_status: Any,
        fs: Any,
        main_program: Optional[Any] = ...,
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
        main_program: Optional[Any] = ...,
        local_cache_path: str = ...,
        ignore_empty: bool = ...,
    ): ...

fleet: Any

class DistributedStrategy(fluid.BuildStrategy):
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
    def __init__(self, optimizer: Any, strategy: Optional[Any] = ...) -> None: ...
    def backward(
        self,
        loss: Any,
        startup_program: Optional[Any] = ...,
        parameter_list: Optional[Any] = ...,
        no_grad_set: Optional[Any] = ...,
        callbacks: Optional[Any] = ...,
    ): ...
    def apply_gradients(self, params_grads: Any): ...

class CollectiveOptimizer(DistributedOptimizer):
    print_config: bool = ...
    def __init__(self, optimizer: Any, strategy: Any = ...) -> None: ...
    def backward(
        self,
        loss: Any,
        startup_program: Optional[Any] = ...,
        parameter_list: Optional[Any] = ...,
        no_grad_set: Optional[Any] = ...,
        callbacks: Optional[Any] = ...,
    ): ...
    def apply_gradients(self, params_grads: Any): ...
    def raiseOptimizeError(self, strategy_name: Any, optimize_name: Any) -> None: ...
    def minimize(
        self,
        loss: Any,
        startup_program: Optional[Any] = ...,
        parameter_list: Optional[Any] = ...,
        no_grad_set: Optional[Any] = ...,
    ): ...
