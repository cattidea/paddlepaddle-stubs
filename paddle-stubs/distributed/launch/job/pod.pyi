from __future__ import annotations

from collections import OrderedDict as OrderedDict
from typing import Any, Optional

from .container import Container as Container
from .status import Status as Status

class PodSepc:
    def __init__(self) -> None: ...

class Pod(PodSepc):
    def __init__(self) -> None: ...
    def failed_container(self): ...
    @property
    def name(self): ...
    @property
    def replicas(self): ...
    @replicas.setter
    def replicas(self, r: Any) -> None: ...
    @property
    def rank(self): ...
    @rank.setter
    def rank(self, r: Any) -> None: ...
    @property
    def restart(self): ...
    @property
    def containers(self): ...
    def add_container(self, c: Any) -> None: ...
    @property
    def init_containers(self): ...
    def add_init_container(self, c: Any) -> None: ...
    @property
    def exit_code(self): ...
    def deploy(self) -> None: ...
    def stop(self, sigint: int = ...) -> None: ...
    def join(self) -> None: ...
    @property
    def status(self): ...
    def reset(self) -> None: ...
    def is_failed(self): ...
    def is_completed(self): ...
    def is_running(self): ...
    def logs(self, idx: Any | None = ...) -> None: ...
    def tail(self, idx: Any | None = ...) -> None: ...
    def watch(self, all_list: Any = ..., any_list: Any = ..., interval: int = ..., timeout: int = ...): ...
