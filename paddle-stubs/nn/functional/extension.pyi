from __future__ import annotations

from typing import Any

from ...fluid import dygraph_utils as dygraph_utils
from ...fluid.data_feeder import check_dtype as check_dtype
from ...fluid.layer_helper import LayerHelper as LayerHelper
from ...fluid.layers.layer_function_generator import templatedoc as templatedoc
from ...fluid.layers.sequence_lod import sequence_mask as sequence_mask
from ...static import Variable as Variable
from ...tensor.creation import assign as assign

def diag_embed(input: Any, offset: int = ..., dim1: int = ..., dim2: int = ...): ...
