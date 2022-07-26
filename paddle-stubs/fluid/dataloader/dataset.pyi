from __future__ import annotations

from typing import Any, Optional

class Dataset:
    def __init__(self) -> None: ...
    def __getitem__(self, idx: Any) -> None: ...
    def __len__(self) -> None: ...

class IterableDataset(Dataset):
    def __init__(self) -> None: ...
    def __iter__(self) -> Any: ...
    def __getitem__(self, idx: Any) -> None: ...
    def __len__(self) -> None: ...

class TensorDataset(Dataset):
    tensors: Any = ...
    def __init__(self, tensors: Any) -> None: ...
    def __getitem__(self, index: Any): ...
    def __len__(self): ...

class ComposeDataset(Dataset):
    datasets: Any = ...
    def __init__(self, datasets: Any) -> None: ...
    def __len__(self): ...
    def __getitem__(self, idx: Any): ...

class ChainDataset(IterableDataset):
    datasets: Any = ...
    def __init__(self, datasets: Any) -> None: ...
    def __iter__(self) -> Any: ...

class Subset(Dataset):
    dataset: Any = ...
    indices: Any = ...
    def __init__(self, dataset: Any, indices: Any) -> None: ...
    def __getitem__(self, idx: Any): ...
    def __len__(self): ...

def random_split(dataset: Any, lengths: Any, generator: Any | None = ...): ...
