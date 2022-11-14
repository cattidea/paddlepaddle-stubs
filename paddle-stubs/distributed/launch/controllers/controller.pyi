from __future__ import annotations

from typing import Any, Optional

from .master import Master as Master

class ControleMode:
    COLLECTIVE: str = ...
    PS: str = ...

class ControllerBase:
    ctx: Any = ...
    master: Any = ...
    job: Any = ...
    pod: Any = ...
    join_server: Any = ...
    def __init__(self, ctx: Any) -> None: ...
    def deploy_pod(self) -> None: ...
    def run(self) -> None: ...
    def watch(self) -> bool: ...
    def stop(self, sigint: Any | None = ...) -> None: ...
    def finalize(self) -> None: ...
    sigint: Any = ...
    def signal_handler(self, sigint: Any, frame: Any) -> None: ...

class Controller(ControllerBase):
    def build_job(self) -> None: ...
    def build_pod(self) -> bool: ...
    def new_container(
        self,
        entrypoint: Any | None = ...,
        envs: Any = ...,
        use_ctx_env: bool = ...,
        out: Any | None = ...,
        err: Any | None = ...,
    ): ...
    def add_container(
        self,
        container: Any | None = ...,
        entrypoint: Any | None = ...,
        envs: Any = ...,
        log_tag: Any | None = ...,
        is_init: bool = ...,
    ) -> None: ...
    def pod_replicas(self): ...
    def save_pod_log(self, info: Any) -> None: ...
