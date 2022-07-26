from __future__ import annotations

from typing import Any, Optional

class Role:
    WORKER: int = ...
    SERVER: int = ...
    XPU: int = ...

class MockBarrier:
    def barrier(self) -> None: ...
    def barrier_all(self) -> None: ...
    def all_reduce(self, obj: Any): ...
    def all_gather(self, obj: Any): ...

class RoleMakerBase:
    def __init__(self) -> None: ...
    def is_worker(self) -> None: ...
    def is_server(self) -> None: ...
    def is_first_worker(self) -> None: ...
    def worker_num(self) -> None: ...
    def role_id(self): ...
    def worker_index(self) -> None: ...
    def server_index(self) -> None: ...
    def get_trainer_endpoints(self): ...
    def get_pserver_endpoints(self): ...
    def to_string(self): ...
    def all_gather(self, input: Any) -> None: ...
    def all_reduce_worker(self, input: Any, output: Any, mode: str = ...) -> None: ...
    def barrier_worker(self) -> None: ...
    def barrier_all(self) -> None: ...

class MPIRoleMaker(RoleMakerBase):
    MPI: Any = ...
    def __init__(self) -> None: ...
    def get_local_ip(self): ...
    def generate_role(self) -> None: ...

class MPISymetricRoleMaker(MPIRoleMaker):
    def __init__(self) -> None: ...
    def all_gather(self, input: Any): ...
    def all_reduce_worker(self, input: Any, output: Any, mode: str = ...) -> None: ...
    def barrier_worker(self) -> None: ...
    def barrier_all(self) -> None: ...
    def is_first_worker(self): ...
    def get_pserver_endpoints(self): ...
    def worker_num(self): ...
    def is_worker(self): ...
    def is_server(self): ...
    def worker_index(self): ...
    def server_index(self): ...
    def generate_role(self) -> None: ...

class PaddleCloudRoleMaker(RoleMakerBase):
    def __init__(self, is_collective: bool = ...) -> None: ...
    def generate_role(self) -> None: ...
    def get_pserver_endpoints(self): ...
    def is_worker(self): ...
    def is_server(self): ...
    def is_first_worker(self): ...
    def worker_index(self): ...
    def server_index(self): ...
    def worker_num(self): ...

class GeneralRoleMaker(RoleMakerBase):
    def __init__(self, **kwargs: Any) -> None: ...
    def generate_role(self) -> None: ...
    def all_gather(self, input: Any): ...
    def all_reduce_worker(self, input: Any, output: Any, mode: str = ...) -> None: ...
    def barrier_worker(self) -> None: ...
    def barrier_all(self) -> None: ...
    def get_local_endpoint(self): ...
    def get_trainer_endpoints(self): ...
    def get_pserver_endpoints(self): ...
    def is_worker(self): ...
    def is_server(self): ...
    def is_first_worker(self): ...
    def worker_index(self): ...
    def server_index(self): ...
    def worker_num(self): ...
    def server_num(self): ...

class HeterRoleMaker(GeneralRoleMaker):
    def generate_role(self) -> None: ...
    def is_xpu(self): ...
    def is_first_xpu(self): ...
    def xpu_num(self): ...

class UserDefinedRoleMaker(RoleMakerBase):
    def __init__(
        self, current_id: int = ..., role: Any = ..., worker_num: int = ..., server_endpoints: Any | None = ...
    ) -> None: ...
    def generate_role(self) -> None: ...
    def is_worker(self): ...
    def is_server(self): ...
    def is_first_worker(self): ...
    def worker_index(self): ...
    def server_index(self): ...
    def worker_num(self): ...

class UserDefinedCollectiveRoleMaker(RoleMakerBase):
    def __init__(self, current_id: int = ..., worker_endpoints: Any | None = ...) -> None: ...
    def generate_role(self) -> None: ...
    def is_worker(self): ...
    def is_first_worker(self): ...
    def worker_index(self): ...
    def worker_num(self): ...
