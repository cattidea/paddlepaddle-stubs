from __future__ import annotations

from typing import Any, Optional

class ParamAttr:
    name: Any = ...
    initializer: Any = ...
    learning_rate: Any = ...
    regularizer: Any = ...
    trainable: Any = ...
    do_model_average: Any = ...
    need_clip: Any = ...
    def __init__(
        self,
        name: Optional[str] = ...,
        initializer: Optional[Any] = ...,
        learning_rate: float = ...,
        regularizer: Optional[Any] = ...,
        trainable: bool = ...,
        do_model_average: bool = ...,
        need_clip: bool = ...,
    ) -> None: ...

class WeightNormParamAttr(ParamAttr):
    params_with_weight_norm: Any = ...
    dim: Any = ...
    def __init__(
        self,
        dim: Optional[Any] = ...,
        name: Optional[str] = ...,
        initializer: Optional[Any] = ...,
        learning_rate: float = ...,
        regularizer: Optional[Any] = ...,
        trainable: bool = ...,
        do_model_average: bool = ...,
        need_clip: bool = ...,
    ) -> None: ...
