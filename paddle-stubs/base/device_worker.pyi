from __future__ import annotations

class DeviceWorker:
    def __init__(self) -> None: ...

class Hogwild(DeviceWorker):
    def __init__(self) -> None: ...

class DownpourLite(DeviceWorker):
    def __init__(self) -> None: ...

class DownpourSGD(DeviceWorker):
    def __init__(self) -> None: ...

class DownpourSGDOPT(DeviceWorker):
    def __init__(self) -> None: ...

class Section(DeviceWorker):
    def __init__(self) -> None: ...

class HeterSection(DeviceWorker):
    def __init__(self) -> None: ...

class DeviceWorkerFactory: ...