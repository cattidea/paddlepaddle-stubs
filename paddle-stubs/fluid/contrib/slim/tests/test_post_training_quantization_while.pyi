from __future__ import annotations

import unittest
from typing import Any

class TestPostTrainingQuantization(unittest.TestCase):
    download_path: str = ...
    cache_folder: Any = ...
    timestamp: Any = ...
    int8_model_path: Any = ...
    def setUp(self) -> None: ...
    def tearDown(self) -> None: ...
    def cache_unzipping(self, target_folder: Any, zip_path: Any) -> None: ...
    def download_model(self, data_url: Any, data_md5: Any, folder_name: Any): ...
    def run_program(self, model_path: Any, batch_size: Any, infer_iterations: Any): ...
    def generate_quantized_model(
        self,
        model_path: Any,
        algo: str = ...,
        quantizable_op_type: Any = ...,
        is_full_quantize: bool = ...,
        is_use_cache_file: bool = ...,
        is_optimize_model: bool = ...,
        batch_size: int = ...,
        batch_nums: int = ...,
        is_data_loader: bool = ...,
    ) -> None: ...
    def run_test(
        self,
        model_name: Any,
        data_url: Any,
        data_md5: Any,
        algo: Any,
        quantizable_op_type: Any,
        is_full_quantize: Any,
        is_use_cache_file: Any,
        is_optimize_model: Any,
        diff_threshold: Any,
        batch_size: int = ...,
        infer_iterations: int = ...,
        quant_iterations: int = ...,
        is_data_loader: bool = ...,
    ) -> None: ...

class TestPostTrainingKLForWhile(TestPostTrainingQuantization):
    def test_post_training_kl(self) -> None: ...

class TestPostTraininghistForWhile(TestPostTrainingQuantization):
    def test_post_training_hist(self) -> None: ...

class TestPostTrainingmseForWhile(TestPostTrainingQuantization):
    def test_post_training_mse(self) -> None: ...

class TestPostTrainingavgForWhile(TestPostTrainingQuantization):
    def test_post_training_avg(self) -> None: ...

class TestPostTrainingMinMaxForWhile(TestPostTrainingQuantization):
    def test_post_training_min_max(self) -> None: ...

class TestPostTrainingAbsMaxForWhile(TestPostTrainingQuantization):
    def test_post_training_abs_max(self) -> None: ...
