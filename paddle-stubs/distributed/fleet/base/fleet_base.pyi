from __future__ import annotations

from typing import Any, Optional

import paddle

from ..meta_optimizers import HeterParallelOptimizer as HeterParallelOptimizer
from ..meta_optimizers import HybridParallelOptimizer as HybridParallelOptimizer
from ..meta_parallel import PipelineParallel as PipelineParallel
from ..meta_parallel import ShardingParallel as ShardingParallel
from ..meta_parallel import TensorParallel as TensorParallel
from ..meta_parallel import model_parallel_random_seed as model_parallel_random_seed
from .distributed_strategy import DistributedStrategy as DistributedStrategy
from .meta_optimizer_factory import MetaOptimizerFactory as MetaOptimizerFactory
from .role_maker import PaddleCloudRoleMaker as PaddleCloudRoleMaker
from .role_maker import RoleMakerBase as RoleMakerBase
from .role_maker import UserDefinedRoleMaker as UserDefinedRoleMaker
from .runtime_factory import RuntimeFactory as RuntimeFactory
from .strategy_compiler import StrategyCompiler as StrategyCompiler
from .topology import ParallelMode as ParallelMode

class _RecomputeModelWrapper(paddle.nn.Layer):
    def __init__(self, model: Any, segments: int = ..., preserve_rng_state: bool = ...) -> None: ...
    def forward(self, input: Any): ...

def apply_ir_passes(main_program: Any, startup_program: Any, config: Any): ...

inited_runtime_handler: Any
is_non_distributed_check: Any

class Fleet:
    strategy_compiler: Any = ...
    def __init__(self) -> None: ...
    def init(
        self, role_maker: Optional[Any] = ..., is_collective: bool = ..., strategy: Optional[Any] = ...
    ) -> None: ...
    def get_hybrid_communicate_group(self): ...
    def get_hybrid_parallel_topology(self): ...
    def is_first_worker(self): ...
    def worker_index(self): ...
    def worker_num(self): ...
    def node_num(self): ...
    def local_rank(self): ...
    def local_device_ids(self): ...
    def world_device_ids(self): ...
    def is_worker(self): ...
    def worker_endpoints(self, to_string: bool = ...): ...
    def server_num(self): ...
    def server_index(self): ...
    def server_endpoints(self, to_string: bool = ...): ...
    def is_server(self): ...
    def barrier_worker(self) -> None: ...
    def init_worker(self, scopes: Optional[Any] = ...) -> None: ...
    def init_server(self, *args: Any, **kwargs: Any) -> None: ...
    def load_model(self, path: Any, mode: Any) -> None: ...
    def run_server(self) -> None: ...
    def stop_worker(self) -> None: ...
    def save(self, dirname: Any, feed: Any = ..., fetch: Any = ..., **configs: Any) -> None: ...
    def save_inference_model(
        self,
        executor: Any,
        dirname: Any,
        feeded_var_names: Any,
        target_vars: Any,
        main_program: Optional[Any] = ...,
        export_for_deployment: bool = ...,
        mode: int = ...,
    ) -> None: ...
    def save_persistables(
        self, executor: Any, dirname: Any, main_program: Optional[Any] = ..., mode: int = ...
    ) -> None: ...
    def shrink(self, threshold: Optional[Any] = ...) -> None: ...
    user_defined_optimizer: Any = ...
    def distributed_optimizer(self, optimizer: Any, strategy: Optional[Any] = ...): ...
    def distributed_model(self, model: Any): ...
    def state_dict(self): ...
    def set_state_dict(self, state_dict: Any): ...
    def set_lr(self, value: Any): ...
    def get_lr(self): ...
    def step(self): ...
    def clear_grad(self): ...
    def get_loss_scaling(self): ...
    def amp_init(
        self, place: Any, scope: Optional[Any] = ..., test_program: Optional[Any] = ..., use_fp16_test: bool = ...
    ): ...
    def minimize(
        self,
        loss: Any,
        startup_program: Optional[Any] = ...,
        parameter_list: Optional[Any] = ...,
        no_grad_set: Optional[Any] = ...,
    ): ...
    def distributed_scaler(self, scaler: Any): ...
