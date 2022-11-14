from __future__ import annotations

from typing import Any, Optional

def simple_img_conv_pool(
    input: Any,
    num_filters: Any,
    filter_size: Any,
    pool_size: Any,
    pool_stride: Any,
    pool_padding: int = ...,
    pool_type: str = ...,
    global_pooling: bool = ...,
    conv_stride: int = ...,
    conv_padding: int = ...,
    conv_dilation: int = ...,
    conv_groups: int = ...,
    param_attr: Any | None = ...,
    bias_attr: Any | None = ...,
    act: Any | None = ...,
    use_cudnn: bool = ...,
): ...
def img_conv_group(
    input: Any,
    conv_num_filter: Any,
    pool_size: Any,
    conv_padding: int = ...,
    conv_filter_size: int = ...,
    conv_act: Any | None = ...,
    param_attr: Any | None = ...,
    conv_with_batchnorm: bool = ...,
    conv_batchnorm_drop_rate: float = ...,
    pool_stride: int = ...,
    pool_type: str = ...,
    use_cudnn: bool = ...,
): ...
def sequence_conv_pool(
    input: Any,
    num_filters: Any,
    filter_size: Any,
    param_attr: Any | None = ...,
    act: str = ...,
    pool_type: str = ...,
    bias_attr: Any | None = ...,
): ...
def glu(input: Any, dim: int = ...): ...
def scaled_dot_product_attention(
    queries: Any, keys: Any, values: Any, num_heads: int = ..., dropout_rate: float = ...
): ...
