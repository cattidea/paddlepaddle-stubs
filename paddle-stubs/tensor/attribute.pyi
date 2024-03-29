from __future__ import annotations

from typing import Any, Optional

from paddle.static import Variable as Variable

from ..base.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ..base.framework import in_dygraph_mode as in_dygraph_mode
from ..base.layer_helper import LayerHelper as LayerHelper
from ..base.layers import rank as rank
from ..base.layers import shape as shape
from ..framework import core as core

def is_complex(x: Any): ...
def is_floating_point(x: Any): ...
def is_integer(x: Any): ...
def real(x: Any, name: str | None = ...): ...
def imag(x: Any, name: str | None = ...): ...
