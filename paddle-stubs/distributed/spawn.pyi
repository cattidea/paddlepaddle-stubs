from __future__ import annotations

from typing import Any, Optional

class ParallelEnvArgs:
    cluster_node_ips: Any = ...
    node_ip: Any = ...
    use_paddlecloud: Any = ...
    started_port: Any = ...
    print_config: bool = ...
    selected_devices: Any = ...
    def __init__(self) -> None: ...

class MultiprocessContext:
    error_queues: Any = ...
    return_queues: Any = ...
    processes: Any = ...
    sentinels: Any = ...
    def __init__(self, processes: Any, error_queues: Any, return_queues: Any) -> None: ...
    def join(self, timeout: Any | None = ...): ...

def spawn(func: Any, args: Any = ..., nprocs: int = ..., join: bool = ..., daemon: bool = ..., **options: Any): ...
