from __future__ import annotations

from typing import Any, Optional

from .streams import Event as Event
from .streams import Stream as Stream

def current_stream(device: Optional[Any] = ...): ...
def synchronize(device: Optional[Any] = ...): ...
def device_count(): ...
def empty_cache() -> None: ...
def max_memory_allocated(device: Optional[Any] = ...): ...
def max_memory_reserved(device: Optional[Any] = ...): ...
def memory_allocated(device: Optional[Any] = ...): ...
def memory_reserved(device: Optional[Any] = ...): ...
def stream_guard(stream: Any) -> None: ...
def get_device_properties(device: Optional[Any] = ...): ...
def get_device_name(device: Optional[Any] = ...): ...
def get_device_capability(device: Optional[Any] = ...): ...
