from __future__ import annotations

import paddle.base as base

class L1Decay(base.regularizer.L1Decay):
    def __init__(self, coeff: float = ...) -> None: ...

class L2Decay(base.regularizer.L2Decay):
    def __init__(self, coeff: float = ...) -> None: ...
