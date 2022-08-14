from __future__ import annotations

from typing import Any, Optional

def fc(
    x: Any,
    size: Any,
    num_flatten_dims: int = ...,
    weight_attr: Optional[Any] = ...,
    bias_attr: Optional[Any] = ...,
    activation: Optional[Any] = ...,
    name: Optional[str] = ...,
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
    weight_attr: Optional[Any] = ...,
    bias_attr: Optional[Any] = ...,
    name: Optional[str] = ...,
): ...
