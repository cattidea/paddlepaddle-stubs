from __future__ import annotations

from typing import Any, Optional

from ...base.data_feeder import check_dtype as check_dtype
from ...base.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ...base.framework import in_dygraph_mode as in_dygraph_mode
from ...base.layer_helper import LayerHelper as LayerHelper
from ...static import Variable as Variable

def one_hot(x: Any, num_classes: Any, name: str | None = ...): ...
def embedding(x: Any, weight: Any, padding_idx: Any | None = ..., sparse: bool = ..., name: str | None = ...): ...
