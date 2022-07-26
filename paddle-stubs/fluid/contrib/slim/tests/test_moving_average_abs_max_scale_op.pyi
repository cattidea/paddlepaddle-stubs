from __future__ import annotations

import unittest
from typing import Any

def init_data(batch_size: int = ..., img_shape: Any = ..., label_range: int = ...): ...

class TestMovingAverageAbsMaxScaleOp(unittest.TestCase):
    def check_backward(self, use_cuda: Any) -> None: ...
    def test_check_op_times(self) -> None: ...
