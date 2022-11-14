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
        weight_preprocess_layer: Any | None = ...,
        act_preprocess_layer: Any | None = ...,
        weight_quantize_layer: Any | None = ...,
        act_quantize_layer: Any | None = ...,
    ) -> None: ...
    def quantize(self, model: Any): ...
    def save_quantized_model(self, layer: Any, path: Any, input_spec: Any | None = ..., **config: Any) -> None: ...

class ImperativeQuantizeInputs:
    def __init__(
        self,
        quantizable_layer_type: Any = ...,
        weight_quantize_type: str = ...,
        activation_quantize_type: str = ...,
        weight_bits: int = ...,
        activation_bits: int = ...,
        moving_rate: float = ...,
        weight_preprocess_layer: Any | None = ...,
        act_preprocess_layer: Any | None = ...,
        weight_quantize_layer: Any | None = ...,
        act_quantize_layer: Any | None = ...,
    ): ...
    def apply(self, model: Any) -> None: ...

class ImperativeQuantizeOutputs:
    def __init__(self, moving_rate: float = ...) -> None: ...
    def apply(self, model: Any) -> None: ...
    def save_quantized_model(
        self, model: Any, path: Any, input_spec: Any | None = ..., onnx_format: bool = ..., **config: Any
    ) -> None: ...
