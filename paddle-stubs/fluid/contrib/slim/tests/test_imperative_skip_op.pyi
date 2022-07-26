from __future__ import annotations

import unittest

from paddle.fluid.dygraph.nn import Pool2D as Pool2D
from paddle.nn import BatchNorm as BatchNorm
from paddle.nn import Conv2D as Conv2D
from paddle.nn import Linear as Linear
from paddle.nn import Softmax as Softmax
from paddle.nn.layer import LeakyReLU as LeakyReLU
from paddle.nn.layer import ReLU as ReLU
from paddle.nn.layer import ReLU6 as ReLU6
from paddle.nn.layer import Sigmoid as Sigmoid

class TestImperativeOutSclae(unittest.TestCase):
    def func_out_scale_acc(self) -> None: ...
    def test_out_scale_acc(self) -> None: ...
