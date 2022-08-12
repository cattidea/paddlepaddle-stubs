from __future__ import annotations

from typing import Any, Optional

from paddle import in_dynamic_mode as in_dynamic_mode

from ..._typing import Tensor
from ...fluid.data_feeder import check_type as check_type
from ...fluid.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ...fluid.layer_helper import LayerHelper as LayerHelper
from .. import Layer as Layer

class PairwiseDistance(Layer):
    p: Any = ...
    epsilon: Any = ...
    keepdim: Any = ...
    name: Any = ...
    def __init__(
        self,
        p: float = ...,
        epsilon: float = ...,
        keepdim: bool = ...,
        name: Optional[str] = ...,
    ) -> None: ...
    def forward(self, x: Tensor, y: Tensor) -> Tensor: ...
    __call__ = forward
