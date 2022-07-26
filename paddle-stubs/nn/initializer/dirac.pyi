from __future__ import annotations

from typing import Any, Optional

from ... import fluid as fluid
from ...fluid import framework as framework
from ...fluid.core import VarDesc as VarDesc
from ...fluid.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ...fluid.initializer import Initializer as Initializer

class Dirac(Initializer):
    def __init__(self, groups: int = ..., name: Optional[Any] = ...) -> None: ...
    def __call__(self, var: Any, block: Optional[Any] = ...): ...
