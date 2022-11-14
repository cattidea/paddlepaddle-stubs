from __future__ import annotations

from typing import Any, Optional

class DataLoaderBase:
    def __init__(self) -> None: ...
    def __call__(self): ...
    def next(self): ...
    def __iter__(self) -> Any: ...
    def __next__(self) -> None: ...

class AuToTune:
    loader: Any = ...
    max_num_worker: Any = ...
    def __init__(self, loader: Any) -> None: ...
    def __call__(self): ...
    def need_autotune(self): ...
    def get_sub_dataset(self, dataset: Any, batch_size: Any): ...
    def get_autotune_loader(self): ...
    def evaluate_reader_cost(self, reader: Any): ...
    def is_best(self, reader: Any, best_workers: Any, best_time: Any, num_work_boundary: Any): ...

class DataLoader:
    return_list: Any = ...
    collate_fn: Any = ...
    use_buffer_reader: Any = ...
    prefetch_factor: Any = ...
    worker_init_fn: Any = ...
    dataset: Any = ...
    feed_list: Any = ...
    places: Any = ...
    num_workers: Any = ...
    use_shared_memory: Any = ...
    timeout: Any = ...
    dataset_kind: Any = ...
    batch_sampler: Any = ...
    batch_size: Any = ...
    drop_last: Any = ...
    auto_collate_batch: Any = ...
    pin_memory: bool = ...
    def __init__(
        self,
        dataset: Any,
        feed_list: Any | None = ...,
        places: Any | None = ...,
        return_list: bool = ...,
        batch_sampler: Any | None = ...,
        batch_size: int = ...,
        shuffle: bool = ...,
        drop_last: bool = ...,
        collate_fn: Any | None = ...,
        num_workers: int = ...,
        use_buffer_reader: bool = ...,
        prefetch_factor: int = ...,
        use_shared_memory: bool = ...,
        timeout: int = ...,
        worker_init_fn: Any | None = ...,
        persistent_workers: bool = ...,
    ) -> None: ...
    def __len__(self): ...
    def __iter__(self) -> Any: ...
    def __call__(self): ...
    @staticmethod
    def from_generator(
        feed_list: Any | None = ...,
        capacity: Any | None = ...,
        use_double_buffer: bool = ...,
        iterable: bool = ...,
        return_list: bool = ...,
        use_multiprocess: bool = ...,
        drop_last: bool = ...,
    ): ...
    @staticmethod
    def from_dataset(dataset: Any, places: Any, drop_last: bool = ...): ...

class DygraphGeneratorLoader(DataLoaderBase):
    def __init__(
        self,
        feed_list: Any | None = ...,
        capacity: Any | None = ...,
        use_double_buffer: bool = ...,
        iterable: bool = ...,
        return_list: bool = ...,
        use_multiprocess: bool = ...,
    ) -> None: ...
    @property
    def queue(self): ...
    @property
    def iterable(self): ...
    def __iter__(self) -> Any: ...
    def __next__(self): ...
    def set_sample_generator(
        self, reader: Any, batch_size: Any, drop_last: bool = ..., places: Any | None = ...
    ): ...
    def set_sample_list_generator(self, reader: Any, places: Any | None = ...): ...
    def set_batch_generator(self, reader: Any, places: Any | None = ...): ...

class GeneratorLoader(DataLoaderBase):
    def __init__(
        self,
        feed_list: Any | None = ...,
        capacity: Any | None = ...,
        use_double_buffer: bool = ...,
        iterable: bool = ...,
        return_list: bool = ...,
        drop_last: bool = ...,
    ) -> None: ...
    @property
    def queue(self): ...
    @property
    def iterable(self): ...
    def __iter__(self) -> Any: ...
    def __next__(self): ...
    def start(self) -> None: ...
    def reset(self) -> None: ...
    def set_sample_generator(
        self, reader: Any, batch_size: Any, drop_last: bool = ..., places: Any | None = ...
    ): ...
    def set_sample_list_generator(self, reader: Any, places: Any | None = ...): ...
    def set_batch_generator(self, reader: Any, places: Any | None = ...): ...

class PyReader(DataLoaderBase):
    def __init__(
        self,
        feed_list: Any | None = ...,
        capacity: Any | None = ...,
        use_double_buffer: bool = ...,
        iterable: bool = ...,
        return_list: bool = ...,
    ) -> None: ...
    @property
    def queue(self): ...
    @property
    def iterable(self): ...
    def __iter__(self) -> Any: ...
    def __next__(self): ...
    def start(self) -> None: ...
    def reset(self) -> None: ...
    def decorate_sample_generator(
        self, sample_generator: Any, batch_size: Any, drop_last: bool = ..., places: Any | None = ...
    ) -> None: ...
    def decorate_sample_list_generator(self, reader: Any, places: Any | None = ...) -> None: ...
    def decorate_batch_generator(self, reader: Any, places: Any | None = ...) -> None: ...

class DatasetLoader(DataLoaderBase):
    def __init__(self, dataset: Any, places: Any, drop_last: Any) -> None: ...
    def __iter__(self) -> Any: ...
    def __next__(self): ...
