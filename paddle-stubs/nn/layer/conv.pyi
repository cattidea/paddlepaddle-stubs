from __future__ import annotations

from typing import Any

from typing import Literal, TypeAlias

from ..._typing import (
    DataLayout1D,
    DataLayout2D,
    DataLayout3D,
    IntSequence,
    ShapeLike,
    Tensor,
)
from ...framework import ParamAttr
from .. import Layer

PaddingSizeStr: TypeAlias = Literal["valid", "same"]
PaddingMode: TypeAlias = Literal["zeros", "reflect", "replicate", "circular"]

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
        weight_attr: Any | None = ...,
        bias_attr: Any | None = ...,
        data_format: str = ...,
    ) -> None: ...
    def extra_repr(self) -> str: ...

class Conv1D(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | IntSequence,
        stride: int | IntSequence = ...,
        padding: int | IntSequence | PaddingSizeStr = ...,
        dilation: int = ...,
        groups: int = ...,
        padding_mode: str = ...,
        weight_attr: ParamAttr | None = ...,
        bias_attr: ParamAttr | bool | None = ...,
        data_format: DataLayout1D = ...,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    __call__ = forward

class Conv1DTranspose(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | IntSequence,
        stride: int | IntSequence = ...,
        padding: int | IntSequence | PaddingSizeStr = ...,
        output_padding: int | IntSequence = ...,
        groups: int = ...,
        dilation: int = ...,
        weight_attr: ParamAttr | None = ...,
        bias_attr: ParamAttr | bool | None = ...,
        data_format: str = ...,
    ) -> None: ...
    def forward(self, x: Tensor, output_size: ShapeLike | None = ...) -> Tensor: ...
    __call__ = forward

class Conv2D(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | IntSequence,
        stride: int | IntSequence = ...,
        padding: int | IntSequence | PaddingSizeStr = ...,
        dilation: int | IntSequence = ...,
        groups: int = ...,
        padding_mode: PaddingMode = ...,
        weight_attr: ParamAttr | None = ...,
        bias_attr: ParamAttr | bool | None = ...,
        data_format: DataLayout2D = ...,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    __call__ = forward

class Conv2DTranspose(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | IntSequence,
        stride: int | IntSequence = ...,
        padding: int | IntSequence | PaddingSizeStr = ...,
        output_padding: int | IntSequence = ...,
        dilation: int | IntSequence = ...,
        groups: int = ...,
        weight_attr: ParamAttr | None = ...,
        bias_attr: ParamAttr | bool | None = ...,
        data_format: DataLayout2D = ...,
    ) -> None: ...
    def forward(self, x: Tensor, output_size: ShapeLike | None = ...) -> Tensor: ...
    __call__ = forward

class Conv3D(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | IntSequence,
        stride: int | IntSequence = ...,
        padding: int | IntSequence | PaddingSizeStr = ...,
        dilation: int | IntSequence = ...,
        groups: int = ...,
        padding_mode: PaddingMode = ...,
        weight_attr: ParamAttr | None = ...,
        bias_attr: ParamAttr | bool | None = ...,
        data_format: DataLayout3D = ...,
    ) -> None: ...
    def forward(self, x: Tensor) -> Tensor: ...
    __call__ = forward

class Conv3DTranspose(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | IntSequence,
        stride: int | IntSequence = ...,
        padding: int | IntSequence | PaddingSizeStr = ...,
        output_padding: int | IntSequence = ...,
        dilation: int | IntSequence = ...,
        groups: int = ...,
        weight_attr: ParamAttr | None = ...,
        bias_attr: ParamAttr | bool | None = ...,
        data_format: DataLayout3D = ...,
    ) -> None: ...
    def forward(self, x: Tensor, output_size: ShapeLike | None = ...) -> Tensor: ...
    __call__ = forward
