from __future__ import annotations

from typing import Any, Optional

class DatasetBase:
    proto_desc: Any = ...
    dataset: Any = ...
    thread_num: int = ...
    filelist: Any = ...
    use_ps_gpu: bool = ...
    psgpu: Any = ...
    def __init__(self) -> None: ...
    def init(
        self,
        batch_size: int = ...,
        thread_num: int = ...,
        use_var: Any = ...,
        pipe_command: str = ...,
        input_type: int = ...,
        fs_name: str = ...,
        fs_ugi: str = ...,
        download_cmd: str = ...,
    ) -> None: ...
    def set_filelist(self, filelist: Any) -> None: ...

class InMemoryDataset(DatasetBase):
    fleet_send_batch_size: Any = ...
    is_user_set_queue_num: bool = ...
    queue_num: Any = ...
    parse_ins_id: bool = ...
    parse_content: bool = ...
    parse_logkey: bool = ...
    merge_by_sid: bool = ...
    enable_pv_merge: bool = ...
    merge_by_lineid: bool = ...
    fleet_send_sleep_seconds: Any = ...
    def __init__(self) -> None: ...
    def update_settings(self, **kwargs: Any) -> None: ...
    def init(self, **kwargs: Any) -> None: ...
    def set_date(self, date: Any) -> None: ...
    def tdm_sample(
        self,
        tree_name: Any,
        tree_path: Any,
        tdm_layer_counts: Any,
        start_sample_layer: Any,
        with_hierachy: Any,
        seed: Any,
        id_slot: Any,
    ) -> None: ...
    def load_into_memory(self, is_shuffle: bool = ...) -> None: ...
    def preload_into_memory(self, thread_num: Optional[Any] = ...) -> None: ...
    def wait_preload_done(self) -> None: ...
    def local_shuffle(self) -> None: ...
    def global_shuffle(self, fleet: Optional[Any] = ..., thread_num: int = ...) -> None: ...
    def release_memory(self) -> None: ...
    def get_memory_data_size(self, fleet: Optional[Any] = ...): ...
    def get_shuffle_data_size(self, fleet: Optional[Any] = ...): ...
    def slots_shuffle(self, slots: Any) -> None: ...

class QueueDataset(DatasetBase):
    def __init__(self) -> None: ...
    def init(self, **kwargs: Any) -> None: ...

class FileInstantDataset(DatasetBase):
    def __init__(self) -> None: ...
    def init(self, **kwargs: Any) -> None: ...

class BoxPSDataset(InMemoryDataset):
    boxps: Any = ...
    def __init__(self) -> None: ...
    def init(self, **kwargs: Any) -> None: ...
    def set_date(self, date: Any) -> None: ...
    def begin_pass(self) -> None: ...
    def end_pass(self, need_save_delta: Any) -> None: ...
    def wait_preload_done(self) -> None: ...
    def load_into_memory(self) -> None: ...
    def preload_into_memory(self) -> None: ...
    def slots_shuffle(self, slots: Any) -> None: ...
    def set_current_phase(self, current_phase: Any) -> None: ...
    def get_pv_data_size(self): ...
    def preprocess_instance(self) -> None: ...
    def postprocess_instance(self) -> None: ...
