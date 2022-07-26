from __future__ import annotations

import paddle.fluid as fluid

class L1Decay(fluid.regularizer.L1Decay):
    def __init__(self, coeff: float = ...) -> None: ...

class L2Decay(fluid.regularizer.L2Decay):
    def __init__(self, coeff: float = ...) -> None: ...
