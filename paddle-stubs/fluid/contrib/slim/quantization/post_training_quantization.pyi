from __future__ import annotations

from typing import Any, Optional

class PostTrainingQuantization:
    def __init__(
        self,
        executor: Any | None = ...,
        scope: Any | None = ...,
        model_dir: Any | None = ...,
        model_filename: str | None = ...,
        params_filename: str | None = ...,
        batch_generator: Any | None = ...,
        sample_generator: Any | None = ...,
        data_loader: Any | None = ...,
        batch_size: int = ...,
        batch_nums: Any | None = ...,
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
        skip_tensor_list: Any | None = ...,
        cache_dir: Any | None = ...,
    ) -> None: ...
    def quantize(self): ...
    def save_quantized_model(
        self, save_model_path: Any, model_filename: str | None = ..., params_filename: str | None = ...
    ) -> None: ...

class WeightQuantization:
    def __init__(self, model_dir: Any, model_filename: str | None = ..., params_filename: str | None = ...) -> None: ...
    def quantize_weight_to_int(
        self,
        save_model_dir: Any,
        save_model_filename: str | None = ...,
        save_params_filename: str | None = ...,
        quantizable_op_type: Any = ...,
        weight_bits: int = ...,
        weight_quantize_type: str = ...,
        generate_test_model: bool = ...,
        threshold_rate: float = ...,
    ) -> None: ...
    def convert_weight_to_fp16(self, save_model_dir: Any) -> None: ...
