from __future__ import annotations

from typing import Any, Optional

def fc(
    x: Any,
    size: Any,
    num_flatten_dims: int = ...,
    weight_attr: Any | None = ...,
    bias_attr: Any | None = ...,
    activation: Any | None = ...,
    name: str | None = ...,
): ...
def deform_conv2d(
    x: Any,
    offset: Any,
    mask: Any,
    num_filters: Any,
    filter_size: Any,
    stride: int = ...,
    padding: int = ...,
    dilation: int = ...,
    groups: int = ...,
    deformable_groups: int = ...,
    im2col_step: int = ...,
    weight_attr: Any | None = ...,
    bias_attr: Any | None = ...,
    name: str | None = ...,
): ...
