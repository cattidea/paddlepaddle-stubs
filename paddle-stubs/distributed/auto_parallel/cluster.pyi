from __future__ import annotations

from enum import IntEnum
from typing import Any

class DeviceType(IntEnum):
    UNKNOWN = ...
    CPU = ...
    GPU = ...
    XPU = ...
    NPU = ...
    DCU = ...
    NIC = ...

class LinkType(IntEnum):
    UNKNOWN = ...
    LOC = ...
    SYS = ...
    PHB = ...
    PIX = ...
    PIB = ...
    NVL = ...
    NVB = ...
    NET = ...

class Device:
    def __init__(self, global_id: Any, local_id: Any, machine: Any) -> None: ...
    @property
    def global_id(self): ...
    @global_id.setter
    def global_id(self, value: Any) -> None: ...
    @property
    def local_id(self): ...
    @local_id.setter
    def local_id(self, value: Any) -> None: ...
    @property
    def machine(self): ...
    @machine.setter
    def machine(self, value: Any) -> None: ...
    @property
    def type(self): ...
    @type.setter
    def type(self, value: Any) -> None: ...
    @property
    def model(self): ...
    @model.setter
    def model(self, value: Any) -> None: ...
    @property
    def dp_gflops(self): ...
    @dp_gflops.setter
    def dp_gflops(self, value: Any) -> None: ...
    @property
    def sp_gflops(self): ...
    @sp_gflops.setter
    def sp_gflops(self, value: Any) -> None: ...
    @property
    def memory(self): ...
    @memory.setter
    def memory(self, value: Any) -> None: ...

class Link:
    def __init__(self, source: Any, target: Any) -> None: ...
    @property
    def source(self): ...
    @source.setter
    def source(self, value: Any) -> None: ...
    @property
    def target(self): ...
    @target.setter
    def target(self, value: Any) -> None: ...
    @property
    def type(self): ...
    @type.setter
    def type(self, value: Any) -> None: ...
    @property
    def bandwidth(self): ...
    @bandwidth.setter
    def bandwidth(self, value: Any) -> None: ...
    @property
    def latency(self): ...
    @latency.setter
    def latency(self, value: Any) -> None: ...

class Machine:
    def __init__(self, id: Any) -> None: ...
    @property
    def id(self): ...
    @id.setter
    def id(self, value: Any) -> None: ...
    @property
    def hostname(self): ...
    @hostname.setter
    def hostname(self, value: Any) -> None: ...
    @property
    def addr(self): ...
    @addr.setter
    def addr(self, value: Any) -> None: ...
    @property
    def port(self): ...
    @port.setter
    def port(self, value: Any) -> None: ...
    @property
    def devices(self): ...
    @property
    def links(self): ...
    def add_device(self, device: Any) -> None: ...
    def add_link(self, link: Any) -> None: ...

class Cluster:
    def __init__(self) -> None: ...
    @property
    def machines(self): ...
    def add_machine(self, machine: Any) -> None: ...
    def add_device(self, device: Any) -> None: ...
    def add_link(self, link: Any) -> None: ...
    def get_device(self, device_global_id: Any): ...
    def build_from_file(self, json_file_path: Any) -> None: ...
    def get_all_devices(self, device_type: Any): ...
