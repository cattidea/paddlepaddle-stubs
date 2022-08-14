from __future__ import annotations

from typing import Any, Optional

class PostTrainingQuantization:
    def __init__(
        self,
        executor: Optional[Any] = ...,
        scope: Optional[Any] = ...,
        model_dir: Optional[Any] = ...,
        model_filename: Optional[str] = ...,
        params_filename: Optional[str] = ...,
        batch_generator: Optional[Any] = ...,
        sample_generator: Optional[Any] = ...,
        data_loader: Optional[Any] = ...,
        batch_size: int = ...,
        batch_nums: Optional[Any] = ...,
        algo: str = ...,
        hist_percent: float = ...,
        quantizable_op_type: Any = ...,
        round_type: str = ...,
        learning_rate: float = ...,
        is_full_quantize: bool = ...,
        bias_correction: bool = ...,
        activation_bits: int = ...,
        weight_bits: int = ...,
        activation_quantize_type: str = ...,
        weight_quantize_type: str = ...,
        onnx_format: bool = ...,
        optimize_model: bool = ...,
        is_use_cache_file: bool = ...,
        skip_tensor_list: Optional[Any] = ...,
        cache_dir: Optional[Any] = ...,
    ) -> None: ...
    def quantize(self): ...
    def save_quantized_model(
        self, save_model_path: Any, model_filename: Optional[str] = ..., params_filename: Optional[str] = ...
    ) -> None: ...

class WeightQuantization:
    def __init__(
        self, model_dir: Any, model_filename: Optional[str] = ..., params_filename: Optional[str] = ...
    ) -> None: ...
    def quantize_weight_to_int(
        self,
        save_model_dir: Any,
        save_model_filename: Optional[str] = ...,
        save_params_filename: Optional[str] = ...,
        quantizable_op_type: Any = ...,
        weight_bits: int = ...,
        weight_quantize_type: str = ...,
        generate_test_model: bool = ...,
        threshold_rate: float = ...,
    ) -> None: ...
    def convert_weight_to_fp16(self, save_model_dir: Any) -> None: ...
