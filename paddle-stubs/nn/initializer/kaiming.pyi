from __future__ import annotations

from typing import Optional

from ...fluid.initializer import MSRAInitializer

class KaimingNormal(MSRAInitializer):
    def __init__(
        self,
        fan_in: float | None = ...,
        negative_slope: float = ...,
        nonlinearity: str = ...,
    ) -> None: ...

class KaimingUniform(MSRAInitializer):
    def __init__(
        self,
        fan_in: float | None = ...,
        negative_slope: float = ...,
        nonlinearity: str = ...,
    ) -> None: ...
