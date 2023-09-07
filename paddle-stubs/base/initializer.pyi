from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from .._typing import Tensor

class Initializer:
    def __init__(self) -> None: ...
    def __call__(self, param: Tensor, block: Any | None = ...) -> Tensor: ...

class ConstantInitializer(Initializer):
    def __init__(self, value: float = ..., force_cpu: bool = ...) -> None: ...
    def __call__(self, var: Tensor, block: Any | None = ...) -> Tensor: ...

class UniformInitializer(Initializer):
    def __init__(
        self,
        low: float = ...,
        high: float = ...,
        seed: int = ...,
        diag_num: int = ...,
        diag_step: int = ...,
        diag_val: float = ...,
    ) -> None: ...
    def __call__(self, var: Tensor, block: Any | None = ...) -> Tensor: ...

class NormalInitializer(Initializer):
    def __init__(self, loc: float = ..., scale: float = ..., seed: int = ...) -> None: ...
    def __call__(self, var: Tensor, block: Any | None = ...) -> Tensor: ...

class TruncatedNormalInitializer(Initializer):
    def __init__(self, loc: float = ..., scale: float = ..., seed: int = ...) -> None: ...
    def __call__(self, var: Tensor, block: Any | None = ...) -> Tensor: ...

class XavierInitializer(Initializer):
    def __init__(
        self, uniform: bool = ..., fan_in: float | None = ..., fan_out: Any | None = ..., seed: int = ...
    ) -> None: ...
    def __call__(self, var: Tensor, block: Any | None = ...) -> Tensor: ...

class MSRAInitializer(Initializer):
    def __init__(
        self,
        uniform: bool = ...,
        fan_in: float | None = ...,
        seed: int = ...,
        negative_slope: int = ...,
        nonlinearity: str = ...,
    ) -> None: ...
    def __call__(self, var: Tensor, block: Any | None = ...) -> Tensor: ...

class BilinearInitializer(Initializer):
    def __init__(self) -> None: ...
    def __call__(self, var: Tensor, block: Any | None = ...) -> Tensor: ...

class NumpyArrayInitializer(Initializer):
    def __init__(self, value: npt.NDArray[Any]) -> None: ...
    def __call__(self, var: Tensor, block: Any | None = ...) -> Tensor: ...

def set_global_initializer(weight_init: Initializer, bias_init: Initializer | None = ...) -> None: ...

Constant = ConstantInitializer
Uniform = UniformInitializer
Normal = NormalInitializer
TruncatedNormal = TruncatedNormalInitializer
Xavier = XavierInitializer
MSRA = MSRAInitializer
Bilinear = BilinearInitializer
