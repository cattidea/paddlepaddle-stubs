from __future__ import annotations

from typing import Any, Optional

from ...fluid import framework as framework
from ...fluid.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ...fluid.initializer import Initializer as Initializer
from ...tensor import diag as diag
from ...tensor import qr as qr
from ...tensor import reshape as reshape
from ...tensor import sign as sign
from ...tensor import transpose as transpose

class Orthogonal(Initializer):
    def __init__(self, gain: float = ..., name: Optional[Any] = ...) -> None: ...
    def __call__(self, var: Any, block: Optional[Any] = ...): ...
