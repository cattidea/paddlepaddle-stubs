from __future__ import annotations

from typing import Any

from ..ps.utils.public import *
from .pass_base import PassBase as PassBase
from .pass_base import register_pass as register_pass

class AppendSendOpsPass(PassBase):
    def __init__(self) -> None: ...

class DistributedOpsPass(PassBase):
    w_2_table_id: Any = ...
    emb_size: Any = ...
    def __init__(self) -> None: ...

class DeleteOptimizesPass(PassBase):
    def __init__(self) -> None: ...

class DeleteExtraOptimizerPass(PassBase):
    def __init__(self) -> None: ...

class FakeInitOpsPass(PassBase):
    def __init__(self) -> None: ...

class PsGpuPass(PassBase):
    def __init__(self) -> None: ...

class PsTranspilePass(PassBase):
    def __init__(self) -> None: ...

class SplitHeterWorkerOpsPass(PassBase):
    def __init__(self) -> None: ...

class SplitTrainerOpsPass(PassBase):
    def __init__(self) -> None: ...

class SetHeterPipelineOptPass(PassBase):
    def __init__(self) -> None: ...
