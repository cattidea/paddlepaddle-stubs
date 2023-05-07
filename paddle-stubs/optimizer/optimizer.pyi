from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from typing_extensions import NotRequired, TypeAlias, TypedDict

from .._typing import Tensor
from ..callbacks import Callback
from ..fluid.clip import GradientClipBase
from ..fluid.framework import Program
from ..fluid.regularizer import WeightDecayRegularizer
from .lr import LRScheduler as LRScheduler

OptimizerStateDict: TypeAlias = dict[str, Tensor]

class ParameterConfig(TypedDict):
    params: Sequence[Tensor]
    weight_decay: NotRequired[float | WeightDecayRegularizer | None]
    learning_rate: NotRequired[float | Tensor | LRScheduler | None]

class Optimizer:
    regularization: Any = ...
    helper: Any = ...
    clear_gradients: Any = ...
    def __init__(
        self,
        learning_rate: float | LRScheduler,
        parameters: list[Tensor] | tuple[Tensor, ...] | None = ...,
        weight_decay: float | WeightDecayRegularizer | None = ...,
        grad_clip: GradientClipBase | None = ...,
        name: str | None = ...,
    ) -> None: ...
    def state_dict(self) -> OptimizerStateDict: ...
    def set_state_dict(self, state_dict: OptimizerStateDict) -> None: ...
    def get_opti_var_name_list(self) -> list[str]: ...
    def set_lr(self, value: float) -> None: ...
    def get_lr(self) -> float: ...
    def backward(
        self,
        loss: Tensor,
        startup_program: Program | None = ...,
        parameters: list[Tensor] | list[str] | None = ...,
        no_grad_set: set[Tensor] | set[str] | None = ...,
        callbacks: list[Callback] | None = ...,
    ) -> list[tuple[Tensor, Tensor]]: ...
    def apply_gradients(self, params_grads: list[tuple[Tensor, Tensor]]) -> list[Any]: ...  # TODO: list[op]
    def append_regularization_ops(
        self,
        parameters_and_grads: list[tuple[Tensor, Tensor]],
        regularization: Any | None = ...,  # TODO: WeightDecayRegularizer?
    ) -> list[tuple[Tensor, Tensor]]: ...
    def clear_grad(self, set_to_zero: bool = ...) -> None: ...
    def minimize(
        self,
        loss: Tensor,
        startup_program: Program | None = ...,
        parameters: list[Tensor] | list[str] | None = ...,
        no_grad_set: set[Tensor] | set[str] | None = ...,
    ) -> list[tuple[Tensor, Tensor]]: ...
    def step(self) -> None: ...
