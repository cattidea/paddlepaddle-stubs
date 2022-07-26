from __future__ import annotations

from typing import Any, Optional

from ...device import get_cudnn_version as get_cudnn_version
from ...device import is_compiled_with_cuda as is_compiled_with_cuda
from ...device import is_compiled_with_rocm as is_compiled_with_rocm
from ...fluid.layers import utils as utils
from .. import Layer as Layer
from ..initializer import Normal as Normal

class _ConvNd(Layer):
    output_padding: Any = ...
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        in_channels: Any,
        out_channels: Any,
        kernel_size: Any,
        transposed: Any,
        dims: Any,
        stride: int = ...,
        padding: int = ...,
        padding_mode: str = ...,
        output_padding: int = ...,
        dilation: int = ...,
        groups: int = ...,
        weight_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        data_format: str = ...,
    ): ...
    def extra_repr(self): ...

class Conv1D(_ConvNd):
    def __init__(
        self,
        in_channels: Any,
        out_channels: Any,
        kernel_size: Any,
        stride: int = ...,
        padding: int = ...,
        dilation: int = ...,
        groups: int = ...,
        padding_mode: str = ...,
        weight_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        data_format: str = ...,
    ) -> None: ...
    def forward(self, x: Any): ...

class Conv1DTranspose(_ConvNd):
    def __init__(
        self,
        in_channels: Any,
        out_channels: Any,
        kernel_size: Any,
        stride: int = ...,
        padding: int = ...,
        output_padding: int = ...,
        groups: int = ...,
        dilation: int = ...,
        weight_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        data_format: str = ...,
    ) -> None: ...
    def forward(self, x: Any, output_size: Optional[Any] = ...): ...

class Conv2D(_ConvNd):
    def __init__(
        self,
        in_channels: Any,
        out_channels: Any,
        kernel_size: Any,
        stride: int = ...,
        padding: int = ...,
        dilation: int = ...,
        groups: int = ...,
        padding_mode: str = ...,
        weight_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        data_format: str = ...,
    ) -> None: ...
    def forward(self, x: Any): ...

class Conv2DTranspose(_ConvNd):
    def __init__(
        self,
        in_channels: Any,
        out_channels: Any,
        kernel_size: Any,
        stride: int = ...,
        padding: int = ...,
        output_padding: int = ...,
        dilation: int = ...,
        groups: int = ...,
        weight_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        data_format: str = ...,
    ) -> None: ...
    def forward(self, x: Any, output_size: Optional[Any] = ...): ...

class Conv3D(_ConvNd):
    def __init__(
        self,
        in_channels: Any,
        out_channels: Any,
        kernel_size: Any,
        stride: int = ...,
        padding: int = ...,
        dilation: int = ...,
        groups: int = ...,
        padding_mode: str = ...,
        weight_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        data_format: str = ...,
    ) -> None: ...
    def forward(self, x: Any): ...

class Conv3DTranspose(_ConvNd):
    def __init__(
        self,
        in_channels: Any,
        out_channels: Any,
        kernel_size: Any,
        stride: int = ...,
        padding: int = ...,
        output_padding: int = ...,
        dilation: int = ...,
        groups: int = ...,
        weight_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        data_format: str = ...,
    ) -> None: ...
    def forward(self, x: Any, output_size: Optional[Any] = ...): ...
