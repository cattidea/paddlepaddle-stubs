from __future__ import annotations

from typing import Any, Optional

from paddle.io import Dataset

DATA_URL: str
LABEL_URL: str
SETID_URL: str
DATA_MD5: str
LABEL_MD5: str
SETID_MD5: str
MODE_FLAG_MAP: Any

class Flowers(Dataset):
    backend: Any = ...
    transform: Any = ...
    data_path: Any = ...
    labels: Any = ...
    indexes: Any = ...
    def __init__(
        self,
        data_file: Optional[Any] = ...,
        label_file: Optional[Any] = ...,
        setid_file: Optional[Any] = ...,
        mode: str = ...,
        transform: Optional[Any] = ...,
        download: bool = ...,
        backend: Optional[Any] = ...,
    ) -> None: ...
    def __getitem__(self, idx: Any): ...
    def __len__(self): ...
