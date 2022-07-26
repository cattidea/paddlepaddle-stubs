from __future__ import annotations

import unittest
from typing import Any

class TestPostTrainingQuantization(unittest.TestCase):
    download_path: str = ...
    cache_folder: Any = ...
    root_path: Any = ...
    int8_model_path: Any = ...
    def setUp(self) -> None: ...
    def tearDown(self) -> None: ...
    def cache_unzipping(self, target_folder: Any, zip_path: Any) -> None: ...
    def download_model(self, data_url: Any, data_md5: Any, folder_name: Any): ...
    def get_batch_reader(self, data_path: Any, place: Any): ...
    def get_simple_reader(self, data_path: Any, place: Any): ...
    def run_program(self, model_path: Any, data_path: Any, infer_iterations: Any): ...
    def generate_quantized_model(
        self,
        model_path: Any,
        data_path: Any,
        algo: str = ...,
        round_type: str = ...,
        quantizable_op_type: Any = ...,
        is_full_quantize: bool = ...,
        is_use_cache_file: bool = ...,
        is_optimize_model: bool = ...,
        batch_size: int = ...,
        batch_nums: int = ...,
        onnx_format: bool = ...,
    ) -> None: ...
    def run_test(
        self,
        model_name: Any,
        model_url: Any,
        model_md5: Any,
        data_name: Any,
        data_url: Any,
        data_md5: Any,
        algo: Any,
        round_type: Any,
        quantizable_op_type: Any,
        is_full_quantize: Any,
        is_use_cache_file: Any,
        is_optimize_model: Any,
        diff_threshold: Any,
        infer_iterations: Any,
        quant_iterations: Any,
        onnx_format: bool = ...,
    ) -> None: ...

class TestPostTrainingAvgForLSTM(TestPostTrainingQuantization):
    def test_post_training_avg(self) -> None: ...

class TestPostTrainingAvgForLSTMONNXFormat(TestPostTrainingQuantization):
    def test_post_training_avg_onnx_format(self) -> None: ...
