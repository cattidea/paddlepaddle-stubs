from __future__ import annotations

from typing import Any, Optional

from paddle.io import Dataset

URL: str
MD5: str

class Imdb(Dataset):
    mode: Any = ...
    data_file: Any = ...
    word_idx: Any = ...
    def __init__(
        self, data_file: Optional[Any] = ..., mode: str = ..., cutoff: int = ..., download: bool = ...
    ) -> None: ...
    def __getitem__(self, idx: Any): ...
    def __len__(self): ...
