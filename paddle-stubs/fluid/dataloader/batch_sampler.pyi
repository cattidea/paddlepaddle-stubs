from __future__ import annotations

from typing import Any, Optional

from .sampler import Sampler

class BatchSampler(Sampler):
    sampler: Any = ...
    batch_size: Any = ...
    drop_last: Any = ...
    def __init__(
        self,
        dataset: Any | None = ...,
        sampler: Any | None = ...,
        shuffle: bool = ...,
        batch_size: int = ...,
        drop_last: bool = ...,
    ) -> None: ...
    def __iter__(self) -> Any: ...
    def __len__(self): ...

class _InfiniteIterableSampler:
    dataset: Any = ...
    batch_size: Any = ...
    def __init__(self, dataset: Any, batch_size: int = ...) -> None: ...
    def __iter__(self) -> Any: ...

class DistributedBatchSampler(BatchSampler):
    dataset: Any = ...
    batch_size: Any = ...
    shuffle: Any = ...
    nranks: Any = ...
    local_rank: Any = ...
    drop_last: Any = ...
    epoch: int = ...
    num_samples: Any = ...
    total_size: Any = ...
    def __init__(
        self,
        dataset: Any,
        batch_size: Any,
        num_replicas: Any | None = ...,
        rank: Any | None = ...,
        shuffle: bool = ...,
        drop_last: bool = ...,
    ) -> None: ...
    def __iter__(self) -> Any: ...
    def __len__(self): ...
    def set_epoch(self, epoch: Any) -> None: ...
