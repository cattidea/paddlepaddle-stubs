from __future__ import annotations

from paddle.base.layers.learning_rate_scheduler import (
    piecewise_decay as piecewise_decay,
)
from paddle.framework import core as core
from paddle.optimizer.lr import PiecewiseDecay as PiecewiseDecay

from ..ps.utils.public import *
from .pass_base import PassBase as PassBase
from .pass_base import register_pass as register_pass

class AddLrDecayTablePass(PassBase):
    def __init__(self) -> None: ...

class AddListenAndServPass(PassBase):
    def __init__(self) -> None: ...

class AddRpcGlobalFlagsPass(PassBase):
    def __init__(self) -> None: ...

class AddOptimizerPass(PassBase):
    def __init__(self) -> None: ...

class AddGeoOptimizerPass(PassBase):
    def __init__(self) -> None: ...

class BuildPserverStartupProgramPass(PassBase):
    def __init__(self) -> None: ...

class DeleteUnusedInStartupPass(PassBase):
    def __init__(self) -> None: ...
