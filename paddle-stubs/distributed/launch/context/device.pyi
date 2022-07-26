from __future__ import annotations

from typing import Any, Optional

class DeviceType:
    CPU: str = ...
    GPU: str = ...
    XPU: str = ...
    NPU: str = ...
    MLU: str = ...

class Device:
    def __init__(self, dtype: Any | None = ..., memory: str = ..., labels: str = ...) -> None: ...
    @property
    def dtype(self): ...
    @property
    def count(self): ...
    @property
    def memory(self): ...
    @property
    def labels(self): ...
    @labels.setter
    def labels(self, lbs: Any) -> None: ...
    def get_selected_device_key(self): ...
    def get_selected_devices(self, devices: str = ...): ...
    @classmethod
    def parse_device(cls): ...
    @classmethod
    def detect_device(cls): ...
