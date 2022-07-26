from __future__ import annotations

import unittest
from typing import Any, Optional

class TestPostTrainingQuantization(unittest.TestCase):
    root_path: Any = ...
    int8_model_path: Any = ...
    download_path: str = ...
    cache_folder: Any = ...
    def setUp(self) -> None: ...
    def tearDown(self) -> None: ...
    def cache_unzipping(self, target_folder: Any, zip_path: Any) -> None: ...
    def download_model(self, data_url: Any, data_md5: Any, folder_name: Any): ...
    def run_program(self, model_path: Any, batch_size: Any, infer_iterations: Any): ...
    def generate_quantized_model(
        self,
        model_path: Any,
        algo: str = ...,
        round_type: str = ...,
        quantizable_op_type: Any = ...,
        is_full_quantize: bool = ...,
        is_use_cache_file: bool = ...,
        is_optimize_model: bool = ...,
        batch_size: int = ...,
        batch_nums: int = ...,
        onnx_format: bool = ...,
        skip_tensor_list: Optional[Any] = ...,
    ) -> None: ...
    def run_test(
        self,
        model_name: Any,
        data_url: Any,
        data_md5: Any,
        algo: Any,
        round_type: Any,
        quantizable_op_type: Any,
        is_full_quantize: Any,
        is_use_cache_file: Any,
        is_optimize_model: Any,
        diff_threshold: Any,
        batch_size: int = ...,
        infer_iterations: int = ...,
        quant_iterations: int = ...,
        onnx_format: bool = ...,
        skip_tensor_list: Optional[Any] = ...,
    ) -> None: ...

class TestPostTrainingKLForMnist(TestPostTrainingQuantization):
    def test_post_training_kl(self) -> None: ...

class TestPostTraininghistForMnist(TestPostTrainingQuantization):
    def test_post_training_hist(self) -> None: ...

class TestPostTrainingmseForMnist(TestPostTrainingQuantization):
    def test_post_training_mse(self) -> None: ...

class TestPostTrainingemdForMnist(TestPostTrainingQuantization):
    def test_post_training_mse(self) -> None: ...

class TestPostTrainingavgForMnist(TestPostTrainingQuantization):
    def test_post_training_avg(self) -> None: ...

class TestPostTrainingAbsMaxForMnist(TestPostTrainingQuantization):
    def test_post_training_abs_max(self) -> None: ...

class TestPostTrainingmseAdaroundForMnist(TestPostTrainingQuantization):
    def test_post_training_mse(self) -> None: ...

class TestPostTrainingKLAdaroundForMnist(TestPostTrainingQuantization):
    def test_post_training_kl(self) -> None: ...

class TestPostTrainingmseForMnistONNXFormat(TestPostTrainingQuantization):
    def test_post_training_mse_onnx_format(self) -> None: ...

class TestPostTrainingmseForMnistONNXFormatFullQuant(TestPostTrainingQuantization):
    def test_post_training_mse_onnx_format_full_quant(self) -> None: ...

class TestPostTrainingavgForMnistSkipOP(TestPostTrainingQuantization):
    def test_post_training_avg_skip_op(self) -> None: ...
