from __future__ import annotations

import abc
from typing import Any, Optional

class Fleet(metaclass=abc.ABCMeta):
    __metaclass__: Any = ...
    def __init__(self, mode: Any) -> None: ...
    def is_first_worker(self): ...
    def worker_index(self): ...
    def worker_num(self): ...
    def is_worker(self): ...
    def worker_endpoints(self, to_string: bool = ...): ...
    def server_num(self): ...
    def server_index(self): ...
    def server_endpoints(self, to_string: bool = ...): ...
    def is_server(self): ...
    def is_xpu(self): ...
    def split_files(self, files: Any): ...
    def init(self, role_maker: Any | None = ...) -> None: ...
    def all_reduce_worker(self, input: Any, output: Any) -> None: ...
    def barrier_worker(self) -> None: ...
    @abc.abstractmethod
    def init_worker(self) -> Any: ...
    @abc.abstractmethod
    def init_server(self, model_dir: Any | None = ..., **kwargs: Any) -> Any: ...
    @abc.abstractmethod
    def run_server(self) -> Any: ...
    @abc.abstractmethod
    def stop_worker(self) -> Any: ...
    @abc.abstractmethod
    def distributed_optimizer(self, optimizer: Any, strategy: Any | None = ...) -> Any: ...
    @abc.abstractmethod
    def save_inference_model(
        self,
        executor: Any,
        dirname: Any,
        feeded_var_names: Any,
        target_vars: Any,
        main_program: Any | None = ...,
        export_for_deployment: bool = ...,
    ) -> Any: ...
    @abc.abstractmethod
    def save_persistables(self, executor: Any, dirname: Any, main_program: Any | None = ...) -> Any: ...

class DistributedOptimizer(metaclass=abc.ABCMeta):
    __metaclass__: Any = ...
    def __init__(self, optimizer: Any, strategy: Any | None = ...) -> None: ...
    @abc.abstractmethod
    def backward(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
        callbacks: Any | None = ...,
    ) -> Any: ...
    @abc.abstractmethod
    def apply_gradients(self, params_grads: Any) -> Any: ...
    @abc.abstractmethod
    def minimize(
        self,
        losses: Any,
        scopes: Any | None = ...,
        startup_programs: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
    ) -> Any: ...
