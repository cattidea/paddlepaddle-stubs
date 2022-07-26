from __future__ import annotations

from typing import Any, Optional

class FileSystem:
    fs_client: Any = ...
    def __init__(
        self,
        fs_type: str = ...,
        uri: str = ...,
        user: Any | None = ...,
        passwd: Any | None = ...,
        hadoop_bin: str = ...,
    ) -> None: ...
    def get_desc(self): ...

class MPIHelper:
    comm: Any = ...
    MPI: Any = ...
    def __init__(self) -> None: ...
    def get_rank(self): ...
    def get_size(self): ...
    def get_ip(self): ...
    def get_hostname(self): ...
    def finalize(self) -> None: ...
