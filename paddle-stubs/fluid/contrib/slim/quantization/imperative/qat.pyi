from __future__ import annotations

from typing import Any, Optional

class ImperativeQuantAware:
    fuse_conv_bn: Any = ...
    def __init__(
        self,
        quantizable_layer_type: Any = ...,
        weight_quantize_type: str = ...,
        activation_quantize_type: str = ...,
        weight_bits: int = ...,
        activation_bits: int = ...,
        moving_rate: float = ...,
        fuse_conv_bn: bool = ...,
        weight_preprocess_layer: Optional[Any] = ...,
        act_preprocess_layer: Optional[Any] = ...,
        weight_quantize_layer: Optional[Any] = ...,
        act_quantize_layer: Optional[Any] = ...,
    ) -> None: ...
    def quantize(self, model: Any): ...
    def save_quantized_model(self, layer: Any, path: Any, input_spec: Optional[Any] = ..., **config: Any) -> None: ...

class ImperativeQuantizeInputs:
    def __init__(
        self,
        quantizable_layer_type: Any = ...,
        weight_quantize_type: str = ...,
        activation_quantize_type: str = ...,
        weight_bits: int = ...,
        activation_bits: int = ...,
        moving_rate: float = ...,
        weight_preprocess_layer: Optional[Any] = ...,
        act_preprocess_layer: Optional[Any] = ...,
        weight_quantize_layer: Optional[Any] = ...,
        act_quantize_layer: Optional[Any] = ...,
    ): ...
    def apply(self, model: Any) -> None: ...

class ImperativeQuantizeOutputs:
    def __init__(self, moving_rate: float = ...) -> None: ...
    def apply(self, model: Any) -> None: ...
    def save_quantized_model(
        self, model: Any, path: Any, input_spec: Optional[Any] = ..., onnx_format: bool = ..., **config: Any
    ) -> None: ...
