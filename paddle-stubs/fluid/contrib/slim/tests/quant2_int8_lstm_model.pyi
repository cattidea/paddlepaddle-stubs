from __future__ import annotations

import unittest
from typing import Any, Optional

def parse_args(): ...

class TestLstmModelPTQ(unittest.TestCase):
    def get_warmup_tensor(self, data_path: Any, place: Any): ...
    def set_config(
        self,
        model_path: Any,
        num_threads: Any,
        mkldnn_cache_capacity: Any,
        warmup_data: Optional[Any] = ...,
        use_analysis: bool = ...,
        enable_ptq: bool = ...,
    ): ...
    def run_program(
        self,
        model_path: Any,
        data_path: Any,
        num_threads: Any,
        mkldnn_cache_capacity: Any,
        warmup_iter: Any,
        use_analysis: bool = ...,
        enable_ptq: bool = ...,
    ): ...
    def test_lstm_model(self) -> None: ...
