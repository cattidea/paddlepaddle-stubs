from __future__ import annotations

from typing import Any, Optional

from paddle.io import Dataset

VOC_URL: str
VOC_MD5: str
SET_FILE: str
DATA_FILE: str
LABEL_FILE: str
CACHE_DIR: str
MODE_FLAG_MAP: Any

class VOC2012(Dataset):
    backend: Any = ...
    flag: Any = ...
    data_file: Any = ...
    transform: Any = ...
    dtype: Any = ...
    def __init__(
        self,
        data_file: Any | None = ...,
        mode: str = ...,
        transform: Any | None = ...,
        download: bool = ...,
        backend: Any | None = ...,
    ) -> None: ...
    def __getitem__(self, idx: Any): ...
    def __len__(self): ...
    def __del__(self) -> None: ...
