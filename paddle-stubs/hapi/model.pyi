from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy.typing as npt
from paddle import nn
from typing_extensions import Literal

from .._typing import DTypeLike, DynamicShapeLike, Tensor
from ..callbacks import Callback
from ..io import DataLoader, Dataset
from ..metric import Metric
from ..optimizer import Optimizer
from ..static import InputSpec
from .model_summary import ModelSummary

class Model:
    mode: str = ...
    network: nn.Layer = ...
    stop_training: bool = ...
    def __init__(
        self,
        network: nn.Layer,
        inputs: InputSpec | list[int] | tuple[int, ...] | dict[str, InputSpec] | None = ...,
        labels: InputSpec | list[int] | tuple[int, ...] | dict[str, InputSpec] | None = ...,
    ) -> None: ...
    def train_batch(
        self,
        inputs: npt.NDArray[Any] | Tensor | list[Tensor],
        labels: npt.NDArray[Any] | Tensor | list[Tensor] | None = ...,
        update: bool = ...,
    ) -> list[npt.NDArray[Any]] | tuple[list[npt.NDArray[Any]], list[npt.NDArray[Any]]]: ...
    def eval_batch(
        self,
        inputs: npt.NDArray[Any] | Tensor | list[Tensor],
        labels: npt.NDArray[Any] | Tensor | list[Tensor] | None = ...,
    ) -> list[npt.NDArray[Any]] | tuple[list[npt.NDArray[Any]], list[npt.NDArray[Any]]]: ...
    def predict_batch(
        self,
        inputs: npt.NDArray[Any] | Tensor | list[Tensor],
    ) -> list[npt.NDArray[Any]]: ...
    def save(self, path: str, training: bool = ...) -> None: ...
    def load(self, path: str, skip_mismatch: bool = ..., reset_optimizer: bool = ...) -> None: ...
    def parameters(self, *args: Any, **kwargs: Any) -> list[Tensor]: ...
    def prepare(
        self,
        optimizer: Optimizer | None = ...,
        loss: nn.Layer | None = ...,
        metrics: Metric | list[Metric] | None = ...,
        amp_configs: Literal["O0", "O1", "O2"] | dict[str, Any] | None = ...,
    ) -> None: ...
    def fit(
        self,
        train_data: Iterable[Tensor] | Dataset | DataLoader | None = ...,
        eval_data: Iterable[Tensor] | Dataset | DataLoader | None = ...,
        batch_size: int = ...,
        epochs: int = ...,
        eval_freq: int = ...,
        log_freq: int = ...,
        save_dir: str | None = ...,
        save_freq: int = ...,
        verbose: int = ...,
        drop_last: bool = ...,
        shuffle: bool = ...,
        num_workers: int = ...,
        callbacks: Callback | list[Callback] | None = ...,
        accumulate_grad_batches: int = ...,
        num_iters: int | None = ...,
    ) -> None: ...
    def evaluate(
        self,
        eval_data: Iterable[Tensor] | Dataset | DataLoader | None,
        batch_size: int = ...,
        log_freq: int = ...,
        verbose: int = ...,
        num_workers: int = ...,
        callbacks: Callback | list[Callback] | None = ...,
        num_iters: int | None = ...,
    ) -> dict[str, npt.NDArray[Any]]: ...
    def predict(
        self,
        test_data: Iterable[Tensor] | Dataset | DataLoader | None,
        batch_size: int = ...,
        num_workers: int = ...,
        stack_outputs: bool = ...,
        verbose: int = ...,
        callbacks: Callback | list[Callback] | None = ...,
    ) -> Tensor: ...  # TODO: multiple output model should return list of tensors
    def summary(
        self,
        input_size: DynamicShapeLike | InputSpec | list[DynamicShapeLike | InputSpec] | None = ...,
        dtype: DTypeLike | None = ...,
    ) -> ModelSummary: ...
