from __future__ import annotations

import abc
from typing import Any, Optional

from paddle.io import DataLoader as DataLoader
from paddle.io import DistributedBatchSampler as DistributedBatchSampler

from .utils import to_list as to_list

class DistributedDataLoader(metaclass=abc.ABCMeta):
    dataset: Any = ...
    batch_size: Any = ...
    epochs: Any = ...
    data_parallel_world_size: Any = ...
    data_parallel_rank: Any = ...
    drop_lost: Any = ...
    def __init__(
        self,
        dataset: Any,
        batch_size: int = ...,
        epochs: int = ...,
        data_parallel_world_size: Any | None = ...,
        data_parallel_rank: Any | None = ...,
        drop_last: bool = ...,
    ) -> None: ...
    @abc.abstractmethod
    def __iter__(self) -> Any: ...
    @abc.abstractmethod
    def __next__(self) -> Any: ...

class NonIterableGeneratorLoader(DistributedDataLoader):
    feed_list: Any = ...
    places: Any = ...
    steps_per_epoch: Any = ...
    def __init__(
        self,
        dataset: Any,
        feed_list: Any,
        places: Any,
        batch_size: int = ...,
        epochs: int = ...,
        steps_per_epoch: Any | None = ...,
        data_parallel_world_size: Any | None = ...,
        data_parallel_rank: Any | None = ...,
        drop_last: bool = ...,
        inputs: Any = ...,
    ) -> None: ...
    def __iter__(self) -> Any: ...
    def __next__(self) -> None: ...
