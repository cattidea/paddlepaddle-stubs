from __future__ import annotations

from typing import Any, Iterable, Optional

import numpy as np
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
        inputs: Optional[InputSpec | list[int] | tuple[int, ...] | dict[str, InputSpec]] = ...,
        labels: Optional[InputSpec | list[int] | tuple[int, ...] | dict[str, InputSpec]] = ...,
    ) -> None: ...
    def train_batch(
        self,
        inputs: npt.NDArray[Any] | Tensor | list[Tensor],
        labels: Optional[npt.NDArray[Any] | Tensor | list[Tensor]] = ...,
        update: bool = ...,
    ) -> list[npt.NDArray[Any]] | tuple[list[npt.NDArray[Any]], list[npt.NDArray[Any]]]: ...
    def eval_batch(
        self,
        inputs: npt.NDArray[Any] | Tensor | list[Tensor],
        labels: Optional[npt.NDArray[Any] | Tensor | list[Tensor]] = ...,
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
        optimizer: Optional[Optimizer] = ...,
        loss: Optional[nn.Layer] = ...,
        metrics: Optional[Metric | list[Metric]] = ...,
        amp_configs: Optional[Literal["O0", "O1", "O2"] | dict[str, Any]] = ...,
    ) -> None: ...
    def fit(
        self,
        train_data: Optional[Iterable[Tensor] | Dataset | DataLoader] = ...,
        eval_data: Optional[Iterable[Tensor] | Dataset | DataLoader] = ...,
        batch_size: int = ...,
        epochs: int = ...,
        eval_freq: int = ...,
        log_freq: int = ...,
        save_dir: Optional[str] = ...,
        save_freq: int = ...,
        verbose: int = ...,
        drop_last: bool = ...,
        shuffle: bool = ...,
        num_workers: int = ...,
        callbacks: Optional[Callback | list[Callback]] = ...,
        accumulate_grad_batches: int = ...,
        num_iters: Optional[int] = ...,
    ) -> None: ...
    def evaluate(
        self,
        eval_data: Optional[Iterable[Tensor] | Dataset | DataLoader],
        batch_size: int = ...,
        log_freq: int = ...,
        verbose: int = ...,
        num_workers: int = ...,
        callbacks: Optional[Callback | list[Callback]] = ...,
        num_iters: Optional[int] = ...,
    ) -> dict[str, npt.NDArray[Any]]: ...
    def predict(
        self,
        test_data: Optional[Iterable[Tensor] | Dataset | DataLoader],
        batch_size: int = ...,
        num_workers: int = ...,
        stack_outputs: bool = ...,
        verbose: int = ...,
        callbacks: Optional[Callback | list[Callback]] = ...,
    ) -> Tensor: ...  # TODO: multiple output model should return list of tensors
    def summary(
        self,
        input_size: Optional[DynamicShapeLike | InputSpec | list[DynamicShapeLike | InputSpec]] = ...,
        dtype: Optional[DTypeLike] = ...,
    ) -> ModelSummary: ...
