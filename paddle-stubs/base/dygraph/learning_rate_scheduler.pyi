from __future__ import annotations

from typing import Any, Optional

class LearningRateDecay:
    step_num: Any = ...
    step_size: Any = ...
    dtype: Any = ...
    def __init__(self, begin: int = ..., step: int = ..., dtype: str = ...) -> None: ...
    def __call__(self): ...
    def create_lr_var(self, lr: Any): ...
    def state_dict(self): ...
    def set_state_dict(self, state_dict: Any) -> None: ...
    set_dict: Any = ...
    def step(self) -> None: ...

class PiecewiseDecay(LearningRateDecay):
    boundaries: Any = ...
    values: Any = ...
    vars: Any = ...
    def __init__(self, boundaries: Any, values: Any, begin: Any, step: int = ..., dtype: str = ...) -> None: ...
    def step(self): ...

class NaturalExpDecay(LearningRateDecay):
    learning_rate: Any = ...
    decay_steps: Any = ...
    decay_rate: Any = ...
    staircase: Any = ...
    def __init__(
        self,
        learning_rate: Any,
        decay_steps: Any,
        decay_rate: Any,
        staircase: bool = ...,
        begin: int = ...,
        step: int = ...,
        dtype: str = ...,
    ) -> None: ...
    def step(self): ...

class ExponentialDecay(LearningRateDecay):
    learning_rate: Any = ...
    decay_steps: Any = ...
    decay_rate: Any = ...
    staircase: Any = ...
    def __init__(
        self,
        learning_rate: Any,
        decay_steps: Any,
        decay_rate: Any,
        staircase: bool = ...,
        begin: int = ...,
        step: int = ...,
        dtype: str = ...,
    ) -> None: ...
    def step(self): ...

class InverseTimeDecay(LearningRateDecay):
    learning_rate: Any = ...
    decay_steps: Any = ...
    decay_rate: Any = ...
    staircase: Any = ...
    def __init__(
        self,
        learning_rate: Any,
        decay_steps: Any,
        decay_rate: Any,
        staircase: bool = ...,
        begin: int = ...,
        step: int = ...,
        dtype: str = ...,
    ) -> None: ...
    def step(self): ...

class PolynomialDecay(LearningRateDecay):
    learning_rate: Any = ...
    decay_steps: Any = ...
    end_learning_rate: Any = ...
    power: Any = ...
    cycle: Any = ...
    def __init__(
        self,
        learning_rate: Any,
        decay_steps: Any,
        end_learning_rate: float = ...,
        power: float = ...,
        cycle: bool = ...,
        begin: int = ...,
        step: int = ...,
        dtype: str = ...,
    ) -> None: ...
    def step(self): ...

class CosineDecay(LearningRateDecay):
    learning_rate: Any = ...
    step_each_epoch: Any = ...
    epochs: Any = ...
    def __init__(
        self, learning_rate: Any, step_each_epoch: Any, epochs: Any, begin: int = ..., step: int = ..., dtype: str = ...
    ) -> None: ...
    def step(self): ...

class NoamDecay(LearningRateDecay):
    learning_rate: Any = ...
    d_model: Any = ...
    warmup_steps: Any = ...
    def __init__(
        self,
        d_model: Any,
        warmup_steps: Any,
        begin: int = ...,
        step: int = ...,
        dtype: str = ...,
        learning_rate: float = ...,
    ) -> None: ...
    def step(self): ...

class LinearLrWarmup(LearningRateDecay):
    learning_rate: Any = ...
    warmup_steps: Any = ...
    start_lr: Any = ...
    lr_ratio_before_warmup: Any = ...
    def __init__(
        self,
        learning_rate: Any,
        warmup_steps: Any,
        start_lr: Any,
        end_lr: Any,
        begin: int = ...,
        step: int = ...,
        dtype: str = ...,
    ) -> None: ...
    def step(self): ...

class ReduceLROnPlateau(LearningRateDecay):
    mode: Any = ...
    decay_rate: Any = ...
    threshold_mode: Any = ...
    learning_rate: Any = ...
    verbose: Any = ...
    patience: Any = ...
    threshold: Any = ...
    cooldown: Any = ...
    min_lr: Any = ...
    eps: Any = ...
    cooldown_counter: int = ...
    best_loss: Any = ...
    num_bad_epochs: int = ...
    epoch_num: int = ...
    def __init__(
        self,
        learning_rate: Any,
        mode: str = ...,
        decay_rate: float = ...,
        patience: int = ...,
        verbose: bool = ...,
        threshold: float = ...,
        threshold_mode: str = ...,
        cooldown: int = ...,
        min_lr: int = ...,
        eps: float = ...,
        dtype: str = ...,
    ) -> None: ...
    def __call__(self): ...
    def step(self, loss: Any) -> None: ...

class _LearningRateEpochDecay(LearningRateDecay):
    base_lr: Any = ...
    epoch_num: int = ...
    dtype: Any = ...
    learning_rate: Any = ...
    def __init__(self, learning_rate: Any, dtype: Any | None = ...) -> None: ...
    def __call__(self): ...
    def epoch(self, epoch: Any | None = ...) -> None: ...
    def get_lr(self) -> None: ...

class StepDecay(_LearningRateEpochDecay):
    step_size: Any = ...
    decay_rate: Any = ...
    def __init__(self, learning_rate: Any, step_size: Any, decay_rate: float = ...) -> None: ...
    def get_lr(self): ...

class MultiStepDecay(_LearningRateEpochDecay):
    milestones: Any = ...
    decay_rate: Any = ...
    def __init__(self, learning_rate: Any, milestones: Any, decay_rate: float = ...) -> None: ...
    def get_lr(self): ...

class LambdaDecay(_LearningRateEpochDecay):
    lr_lambda: Any = ...
    def __init__(self, learning_rate: Any, lr_lambda: Any) -> None: ...
    def get_lr(self): ...