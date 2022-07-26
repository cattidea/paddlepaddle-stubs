from __future__ import annotations

from typing import Any, Optional

class QuantizationTransformPass:
    create_var_map: Any = ...
    create_op_map: Any = ...
    def __init__(
        self,
        scope: Optional[Any] = ...,
        place: Optional[Any] = ...,
        weight_bits: int = ...,
        activation_bits: int = ...,
        activation_quantize_type: str = ...,
        weight_quantize_type: str = ...,
        window_size: int = ...,
        moving_rate: float = ...,
        skip_pattern: Any = ...,
        quantizable_op_type: Any = ...,
        weight_quantize_func: Optional[Any] = ...,
        act_quantize_func: Optional[Any] = ...,
        weight_preprocess_func: Optional[Any] = ...,
        act_preprocess_func: Optional[Any] = ...,
        optimizer_func: Optional[Any] = ...,
        executor: Optional[Any] = ...,
    ) -> None: ...
    def apply(self, graph: Any): ...

class QuantizationFreezePass:
    def __init__(
        self,
        scope: Any,
        place: Any,
        bias_correction: bool = ...,
        weight_bits: int = ...,
        activation_bits: int = ...,
        round_type: str = ...,
        weight_quantize_type: str = ...,
        quantizable_op_type: Optional[Any] = ...,
    ) -> None: ...
    def apply(self, graph: Any): ...

class ConvertToInt8Pass:
    def __init__(self, scope: Any, place: Any, quantizable_op_type: Optional[Any] = ...) -> None: ...
    def apply(self, graph: Any): ...

class TransformForMobilePass:
    def __init__(self) -> None: ...
    def apply(self, graph: Any): ...

class OutScaleForTrainingPass:
    def __init__(self, scope: Optional[Any] = ..., place: Optional[Any] = ..., moving_rate: float = ...) -> None: ...
    def apply(self, graph: Any): ...

class OutScaleForInferencePass:
    def __init__(self, scope: Optional[Any] = ...) -> None: ...
    def apply(self, graph: Any): ...

class AddQuantDequantPass:
    def __init__(
        self,
        scope: Optional[Any] = ...,
        place: Optional[Any] = ...,
        moving_rate: float = ...,
        quant_bits: int = ...,
        skip_pattern: Any = ...,
        quantizable_op_type: Any = ...,
        is_full_quantized: bool = ...,
    ) -> None: ...
    def apply(self, graph: Any): ...

class InsertQuantizeLinear:
    quant_bits: Any = ...
    quant_axis: Any = ...
    channel_wise: Any = ...
    def __init__(
        self,
        place: Any,
        scope: Any,
        quant_bits: int = ...,
        quant_axis: int = ...,
        channel_wise: bool = ...,
        is_test: bool = ...,
    ) -> None: ...
    def insert_quant_op(self, graph: Any, var_node: Any): ...
    def insert_dequant_op(self, graph: Any, var_node: Any, scale_var_node: Any): ...

class QuantizationTransformPassV2:
    create_var_map: Any = ...
    create_op_map: Any = ...
    dequantized_vars: Any = ...
    persistable_vars: Any = ...
    processed_vars: Any = ...
    def __init__(
        self,
        scope: Optional[Any] = ...,
        place: Optional[Any] = ...,
        weight_bits: int = ...,
        activation_bits: int = ...,
        activation_quantize_type: str = ...,
        weight_quantize_type: str = ...,
        window_size: int = ...,
        moving_rate: float = ...,
        skip_pattern: Any = ...,
        quantizable_op_type: Any = ...,
        weight_quantize_func: Optional[Any] = ...,
        act_quantize_func: Optional[Any] = ...,
        weight_preprocess_func: Optional[Any] = ...,
        act_preprocess_func: Optional[Any] = ...,
        optimizer_func: Optional[Any] = ...,
        executor: Optional[Any] = ...,
    ) -> None: ...
    def apply(self, graph: Any): ...

class AddQuantDequantPassV2:
    persistable_vars: Any = ...
    def __init__(
        self,
        scope: Optional[Any] = ...,
        place: Optional[Any] = ...,
        moving_rate: float = ...,
        quant_bits: int = ...,
        skip_pattern: Any = ...,
        quantizable_op_type: Any = ...,
        is_full_quantized: bool = ...,
    ) -> None: ...
    def apply(self, graph: Any): ...

class ReplaceFakeQuantDequantPass:
    def __init__(self, scope: Any, place: Any) -> None: ...
    def apply(self, graph: Any): ...

class QuantWeightPass:
    def __init__(
        self, scope: Any, place: Any, bias_correction: bool = ..., quant_bits: int = ..., save_int_weight: bool = ...
    ) -> None: ...
    def apply(self, graph: Any) -> None: ...
