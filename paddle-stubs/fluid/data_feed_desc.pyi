from __future__ import annotations

from typing import Any

class DataFeedDesc:
    proto_desc: Any = ...
    def __init__(self, proto_file: Any) -> None: ...
    def set_batch_size(self, batch_size: Any) -> None: ...
    def set_dense_slots(self, dense_slots_name: Any) -> None: ...
    def set_use_slots(self, use_slots_name: Any) -> None: ...
    def desc(self): ...
