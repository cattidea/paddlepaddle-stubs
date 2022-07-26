from __future__ import annotations

from typing import Any, Optional

from ..fluid import core as core
from ..fluid import framework as framework
from ..fluid import layers as layers
from ..fluid import unique_name as unique_name
from ..fluid.dygraph import no_grad as no_grad
from ..fluid.framework import Variable as Variable
from ..fluid.framework import name_scope as name_scope
from ..fluid.layer_helper import LayerHelper as LayerHelper
from .optimizer import Optimizer as Optimizer

class SGD(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: float = ...,
        parameters: Optional[Any] = ...,
        weight_decay: Optional[Any] = ...,
        grad_clip: Optional[Any] = ...,
        multi_precision: bool = ...,
        name: Optional[Any] = ...,
    ) -> None: ...
