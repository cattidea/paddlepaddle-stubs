from __future__ import annotations

from typing import Any

class TrainerRuntimeConfig:
    mode: Any = ...
    runtime_configs: Any = ...
    def __init__(self) -> None: ...
    def get_communicator_flags(self): ...
    def display(self, configs: Any): ...

class PSLibRuntimeConfig:
    runtime_configs: Any = ...
    def __init__(self) -> None: ...
    def get_runtime_configs(self): ...

class DistributedStrategy:
    debug_opt: Any = ...
    use_ps_gpu: bool = ...
    def __init__(self) -> None: ...
    def set_debug_opt(self, opt_info: Any) -> None: ...
    def get_debug_opt(self): ...
    def get_program_config(self): ...
    def set_program_config(self, config: Any) -> None: ...
    def check_program_config(self) -> None: ...
    def get_trainer_runtime_config(self): ...
    def set_trainer_runtime_config(self, config: Any) -> None: ...
    def check_trainer_runtime_config(self) -> None: ...
    def get_pslib_runtime_config(self): ...
    def set_pslib_runtime_config(self, config: Any) -> None: ...
    def get_server_runtime_config(self): ...
    def set_server_runtime_config(self, config: Any) -> None: ...
    def check_server_runtime_config(self) -> None: ...
    def get_execute_strategy(self): ...
    def set_execute_strategy(self, config: Any) -> None: ...
    def check_execute_strategy(self) -> None: ...
    def get_build_strategy(self): ...
    def set_build_strategy(self, config: Any) -> None: ...
    def check_build_strategy(self) -> None: ...

class SyncStrategy(DistributedStrategy):
    def __init__(self) -> None: ...
    def check_trainer_runtime_config(self) -> None: ...
    def check_program_config(self) -> None: ...
    def check_server_runtime_config(self) -> None: ...
    def check_execute_strategy(self) -> None: ...
    def check_build_strategy(self) -> None: ...

class AsyncStrategy(DistributedStrategy):
    def __init__(self) -> None: ...
    def check_trainer_runtime_config(self) -> None: ...
    def check_program_config(self) -> None: ...
    def check_server_runtime_config(self) -> None: ...
    def check_execute_strategy(self) -> None: ...
    def check_build_strategy(self) -> None: ...

class HalfAsyncStrategy(DistributedStrategy):
    def __init__(self) -> None: ...
    def check_trainer_runtime_config(self) -> None: ...
    def check_program_config(self) -> None: ...
    def check_server_runtime_config(self) -> None: ...
    def check_execute_strategy(self) -> None: ...
    def check_build_strategy(self) -> None: ...

class GeoStrategy(DistributedStrategy):
    def __init__(self, update_frequency: int = ...) -> None: ...
    def check_program_config(self) -> None: ...
    def check_trainer_runtime_config(self) -> None: ...
    def check_server_runtime_config(self) -> None: ...
    def check_execute_strategy(self) -> None: ...
    def check_build_strategy(self) -> None: ...

class StrategyFactory:
    @staticmethod
    def create_sync_strategy(): ...
    @staticmethod
    def create_half_async_strategy(): ...
    @staticmethod
    def create_async_strategy(): ...
    @staticmethod
    def create_geo_strategy(update_frequency: int = ...): ...
