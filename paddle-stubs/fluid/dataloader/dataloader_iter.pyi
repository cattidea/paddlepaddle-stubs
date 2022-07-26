from __future__ import annotations

from typing import Any

from .worker import get_worker_info as get_worker_info

class _DataLoaderIterBase:
    def __init__(self, loader: Any) -> None: ...
    def __iter__(self) -> Any: ...
    def __len__(self): ...

class _DataLoaderIterSingleProcess(_DataLoaderIterBase):
    def __init__(self, loader: Any) -> None: ...
    def __next__(self): ...
    def next(self): ...
    def __del__(self) -> None: ...

class _DataLoaderIterMultiProcess(_DataLoaderIterBase):
    def __init__(self, loader: Any) -> None: ...
    def __del__(self) -> None: ...
    def __next__(self): ...
    def next(self): ...
