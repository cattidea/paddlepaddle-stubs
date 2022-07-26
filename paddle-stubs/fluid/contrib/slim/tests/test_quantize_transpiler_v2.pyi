from __future__ import annotations

import unittest
from typing import Any

def conv_net(img: Any, label: Any): ...

class TestQuantizeProgramPass(unittest.TestCase):
    def quantize_program(
        self,
        use_cuda: Any,
        seed: Any,
        activation_quant_type: str = ...,
        weight_quant_type: str = ...,
        for_ci: bool = ...,
    ): ...
    def test_gpu_1(self) -> None: ...
    def test_gpu_2(self) -> None: ...
    def test_cpu_1(self) -> None: ...
    def test_cpu_2(self) -> None: ...
