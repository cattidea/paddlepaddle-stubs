from __future__ import annotations

from typing import Any, Optional

from ...device import get_cudnn_version as get_cudnn_version
from ...device import is_compiled_with_rocm as is_compiled_with_rocm
from ...fluid import dygraph_utils as dygraph_utils
from ...fluid.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ...fluid.layer_helper import LayerHelper as LayerHelper
from ...static import Variable as Variable

def affine_grid(theta: Any, out_shape: Any, align_corners: bool = ..., name: str | None = ...): ...
def grid_sample(
    x: Any, grid: Any, mode: str = ..., padding_mode: str = ..., align_corners: bool = ..., name: str | None = ...
): ...
def pixel_shuffle(x: Any, upscale_factor: Any, data_format: str = ..., name: str | None = ...): ...
