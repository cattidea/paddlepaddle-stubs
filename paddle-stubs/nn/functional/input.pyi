from __future__ import annotations

from typing import Any, Optional

from ...fluid.data_feeder import check_dtype as check_dtype
from ...fluid.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ...fluid.framework import in_dygraph_mode as in_dygraph_mode
from ...fluid.layer_helper import LayerHelper as LayerHelper
from ...static import Variable as Variable

def one_hot(x: Any, num_classes: Any, name: Optional[Any] = ...): ...
def embedding(x: Any, weight: Any, padding_idx: Optional[Any] = ..., sparse: bool = ..., name: Optional[Any] = ...): ...
