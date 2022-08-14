from __future__ import annotations

from typing import Any, Optional

from paddle.nn import Layer

class FakeQuantAbsMax(Layer):
    def __init__(
        self, name: Optional[str] = ..., quant_bits: int = ..., dtype: str = ..., quant_on_weight: bool = ...
    ) -> None: ...
    def forward(self, input: Any): ...

class FakeQuantMovingAverageAbsMax(Layer):
    def __init__(
        self, name: Optional[str] = ..., moving_rate: float = ..., quant_bits: int = ..., dtype: str = ...
    ) -> None: ...
    def forward(self, input: Any): ...

class FakeQuantChannelWiseAbsMax(Layer):
    def __init__(
        self,
        name: Optional[str] = ...,
        channel_num: Optional[Any] = ...,
        quant_bits: int = ...,
        quant_axis: int = ...,
        dtype: str = ...,
        quant_on_weight: bool = ...,
    ) -> None: ...
    def forward(self, input: Any): ...

class MovingAverageAbsMaxScale(Layer):
    def __init__(self, name: Optional[str] = ..., moving_rate: float = ..., dtype: str = ...) -> None: ...
    def forward(self, input: Any): ...

QuantStub = MovingAverageAbsMaxScale

class QuantizedConv2D(Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        layer: Any,
        weight_bits: int = ...,
        activation_bits: int = ...,
        moving_rate: float = ...,
        weight_quantize_type: str = ...,
        activation_quantize_type: str = ...,
        weight_pre_layer: Optional[Any] = ...,
        act_pre_layer: Optional[Any] = ...,
        weight_quant_layer: Optional[Any] = ...,
        act_quant_layer: Optional[Any] = ...,
    ) -> None: ...
    def forward(self, input: Any): ...

class QuantizedConv2DTranspose(Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        layer: Any,
        weight_bits: int = ...,
        activation_bits: int = ...,
        moving_rate: float = ...,
        weight_quantize_type: str = ...,
        activation_quantize_type: str = ...,
        weight_pre_layer: Optional[Any] = ...,
        act_pre_layer: Optional[Any] = ...,
        weight_quant_layer: Optional[Any] = ...,
        act_quant_layer: Optional[Any] = ...,
    ) -> None: ...
    def forward(self, input: Any, output_size: Optional[Any] = ...): ...

class QuantizedLinear(Layer):
    weight: Any = ...
    bias: Any = ...
    name: Any = ...
    def __init__(
        self,
        layer: Any,
        weight_bits: int = ...,
        activation_bits: int = ...,
        moving_rate: float = ...,
        weight_quantize_type: str = ...,
        activation_quantize_type: str = ...,
        weight_pre_layer: Optional[Any] = ...,
        act_pre_layer: Optional[Any] = ...,
        weight_quant_layer: Optional[Any] = ...,
        act_quant_layer: Optional[Any] = ...,
    ) -> None: ...
    def forward(self, input: Any): ...

class MAOutputScaleLayer(Layer):
    def __init__(
        self, layer: Optional[Any] = ..., moving_rate: float = ..., name: Optional[str] = ..., dtype: str = ...
    ) -> None: ...
    def forward(self, *inputs: Any, **kwargs: Any): ...

class FakeQuantMAOutputScaleLayer(Layer):
    def __init__(
        self,
        layer: Any,
        weight_bits: int = ...,
        activation_bits: int = ...,
        moving_rate: float = ...,
        name: Optional[str] = ...,
        *args: Any,
        **kwargs: Any,
    ) -> None: ...
    def forward(self, *inputs: Any, **kwargs: Any): ...
