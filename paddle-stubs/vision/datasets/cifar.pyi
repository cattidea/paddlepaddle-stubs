from __future__ import annotations

from typing import Any, Optional

from paddle.io import Dataset

URL_PREFIX: str
CIFAR10_URL: Any
CIFAR10_MD5: str
CIFAR100_URL: Any
CIFAR100_MD5: str
MODE_FLAG_MAP: Any

class Cifar10(Dataset):
    mode: Any = ...
    backend: Any = ...
    data_file: Any = ...
    transform: Any = ...
    dtype: Any = ...
    def __init__(
        self,
        data_file: Optional[Any] = ...,
        mode: str = ...,
        transform: Optional[Any] = ...,
        download: bool = ...,
        backend: Optional[Any] = ...,
    ) -> None: ...
    def __getitem__(self, idx: Any): ...
    def __len__(self): ...

class Cifar100(Cifar10):
    def __init__(
        self,
        data_file: Optional[Any] = ...,
        mode: str = ...,
        transform: Optional[Any] = ...,
        download: bool = ...,
        backend: Optional[Any] = ...,
    ) -> None: ...
