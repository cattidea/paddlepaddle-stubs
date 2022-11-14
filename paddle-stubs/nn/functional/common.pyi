from __future__ import annotations

from typing import Any, Optional

from ...fluid import dygraph_utils as dygraph_utils
from ...fluid import layers as layers
from ...fluid.data_feeder import check_dtype as check_dtype
from ...fluid.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ...fluid.framework import in_dygraph_mode as in_dygraph_mode
from ...fluid.layers import unfold as unfold
from ...tensor import clip as clip
from ...tensor import concat as concat
from ...tensor import sqrt as sqrt
from ...tensor import sum as sum
from ...tensor.creation import zeros as zeros
from ...tensor.manipulation import squeeze as squeeze
from ...tensor.manipulation import unsqueeze as unsqueeze

def interpolate(
    x: Any,
    size: Any | None = ...,
    scale_factor: Any | None = ...,
    mode: str = ...,
    align_corners: bool = ...,
    align_mode: int = ...,
    data_format: str = ...,
    name: str | None = ...,
): ...
def upsample(
    x: Any,
    size: Any | None = ...,
    scale_factor: Any | None = ...,
    mode: str = ...,
    align_corners: bool = ...,
    align_mode: int = ...,
    data_format: str = ...,
    name: str | None = ...,
): ...
def bilinear(x1: Any, x2: Any, weight: Any, bias: Any | None = ..., name: str | None = ...): ...
def dropout(
    x: Any, p: float = ..., axis: Any | None = ..., training: bool = ..., mode: str = ..., name: str | None = ...
): ...
def dropout2d(x: Any, p: float = ..., training: bool = ..., data_format: str = ..., name: str | None = ...): ...
def dropout3d(x: Any, p: float = ..., training: bool = ..., data_format: str = ..., name: str | None = ...): ...
def alpha_dropout(x: Any, p: float = ..., training: bool = ..., name: str | None = ...): ...
def pad(x: Any, pad: Any, mode: str = ..., value: int = ..., data_format: str = ..., name: str | None = ...): ...
def zeropad2d(x: Any, padding: Any, data_format: str = ..., name: str | None = ...): ...
def cosine_similarity(x1: Any, x2: Any, axis: int = ..., eps: float = ...): ...
def linear(x: Any, weight: Any, bias: Any | None = ..., name: str | None = ...): ...
def label_smooth(label: Any, prior_dist: Any | None = ..., epsilon: float = ..., name: str | None = ...): ...
def class_center_sample(label: Any, num_classes: Any, num_samples: Any, group: Any | None = ...): ...
def fold(
    x: Any,
    output_sizes: Any,
    kernel_sizes: Any,
    strides: int = ...,
    paddings: int = ...,
    dilations: int = ...,
    name: str | None = ...,
): ...
