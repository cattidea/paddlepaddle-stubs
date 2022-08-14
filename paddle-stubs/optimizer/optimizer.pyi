from __future__ import annotations

from typing import Any, Optional

from typing_extensions import NotRequired, TypedDict

from .._typing import Tensor
from ..callbacks import Callback
from ..fluid.clip import GradientClipBase
from ..fluid.framework import Program
from ..fluid.regularizer import WeightDecayRegularizer
from .lr import LRScheduler as LRScheduler

OptimizerStateDict = dict[str, Tensor]

class ParameterConfig(TypedDict):
    params: list[Tensor]
    weight_decay: NotRequired[Optional[float | WeightDecayRegularizer]]
    learning_rate: NotRequired[Optional[float | Tensor | LRScheduler]]

class Optimizer:
    regularization: Any = ...
    helper: Any = ...
    clear_gradients: Any = ...
    def __init__(
        self,
        learning_rate: float | LRScheduler,
        parameters: Optional[list[Tensor] | tuple[Tensor, ...]] = ...,
        weight_decay: Optional[float | WeightDecayRegularizer] = ...,
        grad_clip: Optional[GradientClipBase] = ...,
        name: Optional[str] = ...,
    ) -> None: ...
    def state_dict(self) -> OptimizerStateDict: ...
    def set_state_dict(self, state_dict: OptimizerStateDict) -> None: ...
    def get_opti_var_name_list(self) -> list[str]: ...
    def set_lr(self, value: float) -> None: ...
    def get_lr(self) -> float: ...
    def backward(
        self,
        loss: Tensor,
        startup_program: Optional[Program] = ...,
        parameters: Optional[list[Tensor] | list[str]] = ...,
        no_grad_set: Optional[set[Tensor] | set[str]] = ...,
        callbacks: Optional[list[Callback]] = ...,
    ) -> list[tuple[Tensor, Tensor]]: ...
    def apply_gradients(self, params_grads: list[tuple[Tensor, Tensor]]) -> list[Any]: ...  # TODO: list[op]
    def append_regularization_ops(
        self,
        parameters_and_grads: list[tuple[Tensor, Tensor]],
        regularization: Optional[Any] = ...,  # TODO: WeightDecayRegularizer?
    ) -> list[tuple[Tensor, Tensor]]: ...
    def clear_grad(self, set_to_zero: bool = ...) -> None: ...
    def minimize(
        self,
        loss: Tensor,
        startup_program: Optional[Program] = ...,
        parameters: Optional[list[Tensor] | list[str]] = ...,
        no_grad_set: Optional[set[Tensor] | set[str]] = ...,
    ) -> list[tuple[Tensor, Tensor]]: ...
    def step(self) -> None: ...
