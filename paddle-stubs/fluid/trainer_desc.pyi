from __future__ import annotations

from typing import Any

class TrainerDesc:
    proto_desc: Any = ...
    def __init__(self) -> None: ...

class MultiTrainer(TrainerDesc):
    def __init__(self) -> None: ...

class DistMultiTrainer(TrainerDesc):
    def __init__(self) -> None: ...

class HeterXpuTrainer(TrainerDesc):
    def __init__(self) -> None: ...

class PSGPUTrainer(TrainerDesc):
    def __init__(self) -> None: ...

class HeterPipelineTrainer(TrainerDesc):
    def __init__(self) -> None: ...

class PipelineTrainer(TrainerDesc):
    def __init__(self) -> None: ...
