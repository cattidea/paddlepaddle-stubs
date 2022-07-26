from __future__ import annotations

from typing import Any

class WeightDecayRegularizer:
    def __init__(self) -> None: ...
    def __call__(self, param: Any, grad: Any, block: Any) -> None: ...

class L2DecayRegularizer(WeightDecayRegularizer):
    def __init__(self, regularization_coeff: float = ...) -> None: ...
    def __call__(self, param: Any, grad: Any, block: Any): ...

class L1DecayRegularizer(WeightDecayRegularizer):
    def __init__(self, regularization_coeff: float = ...) -> None: ...
    def __call__(self, param: Any, grad: Any, block: Any): ...

L1Decay = L1DecayRegularizer
L2Decay = L2DecayRegularizer
