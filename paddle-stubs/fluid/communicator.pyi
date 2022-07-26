from __future__ import annotations

from typing import Any, Optional

class Communicator:
    mode: Any = ...
    envs: Any = ...
    communicator_: Any = ...
    send_ctx_: Any = ...
    recv_ctx_: Any = ...
    def __init__(self, mode: Any, kwargs: Any | None = ..., envs: Any | None = ...) -> None: ...
    def init_with_ctx(
        self, send_ctx: Any, recv_ctx: Any, proto_txt: Any, unit64_hosts: Any, scope: Any | None = ...
    ) -> None: ...
    def create_client_to_client_connection(
        self, pserver_timeout_ms: int = ..., pserver_connect_timeout_ms: int = ..., max_retry: int = ...
    ) -> None: ...
    def get_client_info(self): ...
    def set_clients(self, host_list: Any) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def is_running(self) -> None: ...
    def recv(self) -> None: ...
    def init_params(self, context: Any) -> None: ...
    def pull_dense(self, context: Any) -> None: ...
    def push_sparse_param(self, var_name: Any, table_id: int = ..., scope: Any | None = ...) -> None: ...

class LargeScaleKV:
    scale_kv: Any = ...
    def __init__(self) -> None: ...
    def save(self, varname: Any, dirname: Any) -> None: ...
    def load(self, varname: Any, dirname: Any) -> None: ...
    def size(self, varname: Any): ...

class HeterClient:
    heter_client_: Any = ...
    def __init__(self, endpoint: Any, previous_endpoint: Any, trainer_id: Any) -> None: ...
    def stop(self) -> None: ...
