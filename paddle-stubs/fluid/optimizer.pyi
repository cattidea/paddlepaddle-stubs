from __future__ import annotations

from typing import Any, Optional

class Optimizer:
    regularization: Any = ...
    helper: Any = ...
    def __init__(
        self,
        learning_rate: Any,
        parameter_list: Any | None = ...,
        regularization: Any | None = ...,
        grad_clip: Any | None = ...,
        flatten_param_grads: bool = ...,
        align_size: int = ...,
        name: str | None = ...,
    ): ...
    def state_dict(self): ...
    def set_state_dict(self, state_dict: Any) -> None: ...
    set_dict: Any = ...
    def get_opti_var_name_list(self): ...
    def set_lr(self, value: Any) -> None: ...
    def current_step_lr(self): ...
    def backward(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
        callbacks: Any | None = ...,
    ): ...
    def append_regularization_ops(self, parameters_and_grads: Any, regularization: Any | None = ...): ...
    def flatten_param_grads(self, params_grads: Any): ...
    def apply_gradients(self, params_grads: Any): ...
    def apply_optimize(self, loss: Any, startup_program: Any, params_grads: Any): ...
    def clear_gradients(self) -> None: ...
    def minimize(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
    ): ...

class SGDOptimizer(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: Any,
        parameter_list: Any | None = ...,
        regularization: Any | None = ...,
        grad_clip: Any | None = ...,
        multi_precision: bool = ...,
        name: str | None = ...,
    ) -> None: ...

class MomentumOptimizer(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: Any,
        momentum: Any,
        parameter_list: Any | None = ...,
        use_nesterov: bool = ...,
        regularization: Any | None = ...,
        grad_clip: Any | None = ...,
        name: str | None = ...,
    ) -> None: ...

class DGCMomentumOptimizer(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: Any,
        momentum: Any,
        rampup_begin_step: Any,
        rampup_step: int = ...,
        sparsity: Any = ...,
        parameter_list: Any | None = ...,
        use_nesterov: bool = ...,
        num_trainers: Any | None = ...,
        regularization: Any | None = ...,
        grad_clip: Any | None = ...,
        name: str | None = ...,
    ) -> None: ...
    def apply_gradients(self, params_grads: Any): ...

class LarsMomentumOptimizer(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: Any,
        momentum: Any,
        lars_coeff: float = ...,
        lars_weight_decay: float = ...,
        parameter_list: Any | None = ...,
        regularization: Any | None = ...,
        grad_clip: Any | None = ...,
        name: str | None = ...,
        exclude_from_weight_decay: Any | None = ...,
        epsilon: int = ...,
        multi_precision: bool = ...,
        rescale_grad: float = ...,
    ) -> None: ...

class AdagradOptimizer(Optimizer):
    type: str = ...
    initial_accumulator_value: Any = ...
    def __init__(
        self,
        learning_rate: Any,
        epsilon: float = ...,
        parameter_list: Any | None = ...,
        regularization: Any | None = ...,
        grad_clip: Any | None = ...,
        name: str | None = ...,
        initial_accumulator_value: float = ...,
    ) -> None: ...

class AdamOptimizer(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: float = ...,
        beta1: float = ...,
        beta2: float = ...,
        epsilon: float = ...,
        parameter_list: Any | None = ...,
        regularization: Any | None = ...,
        grad_clip: Any | None = ...,
        name: str | None = ...,
        lazy_mode: bool = ...,
        use_global_beta_pow: bool = ...,
        flatten_param_grads: bool = ...,
        align_size: int = ...,
    ) -> None: ...

class AdamaxOptimizer(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: float = ...,
        beta1: float = ...,
        beta2: float = ...,
        epsilon: float = ...,
        parameter_list: Any | None = ...,
        regularization: Any | None = ...,
        grad_clip: Any | None = ...,
        name: str | None = ...,
    ) -> None: ...

class DpsgdOptimizer(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: float = ...,
        clip: float = ...,
        batch_size: float = ...,
        sigma: float = ...,
        parameter_list: Any | None = ...,
    ) -> None: ...

class DecayedAdagradOptimizer(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: Any,
        decay: float = ...,
        epsilon: float = ...,
        parameter_list: Any | None = ...,
        regularization: Any | None = ...,
        grad_clip: Any | None = ...,
        name: str | None = ...,
    ) -> None: ...

class AdadeltaOptimizer(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: Any,
        epsilon: float = ...,
        rho: float = ...,
        parameter_list: Any | None = ...,
        regularization: Any | None = ...,
        grad_clip: Any | None = ...,
        name: str | None = ...,
    ) -> None: ...

class RMSPropOptimizer(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: Any,
        rho: float = ...,
        epsilon: float = ...,
        momentum: float = ...,
        centered: bool = ...,
        parameter_list: Any | None = ...,
        regularization: Any | None = ...,
        grad_clip: Any | None = ...,
        name: str | None = ...,
    ) -> None: ...

class FtrlOptimizer(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: Any,
        l1: float = ...,
        l2: float = ...,
        lr_power: Any = ...,
        parameter_list: Any | None = ...,
        regularization: Any | None = ...,
        grad_clip: Any | None = ...,
        name: str | None = ...,
    ) -> None: ...

class LambOptimizer(AdamOptimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: float = ...,
        lamb_weight_decay: float = ...,
        beta1: float = ...,
        beta2: float = ...,
        epsilon: float = ...,
        parameter_list: Any | None = ...,
        regularization: Any | None = ...,
        grad_clip: Any | None = ...,
        exclude_from_weight_decay_fn: Any | None = ...,
        name: str | None = ...,
    ) -> None: ...

SGD = SGDOptimizer
Momentum = MomentumOptimizer
Adagrad = AdagradOptimizer
Adam = AdamOptimizer
Adamax = AdamaxOptimizer
Dpsgd = DpsgdOptimizer
DecayedAdagrad = DecayedAdagradOptimizer
Adadelta = AdadeltaOptimizer
RMSProp = RMSPropOptimizer
Ftrl = FtrlOptimizer
LarsMomentum = LarsMomentumOptimizer
Lamb = LambOptimizer

class ModelAverage(Optimizer):
    average_window: Any = ...
    min_average_window: Any = ...
    max_average_window: Any = ...
    params_grads: Any = ...
    apply_program: Any = ...
    restore_program: Any = ...
    def __init__(
        self,
        average_window_rate: Any,
        min_average_window: int = ...,
        max_average_window: int = ...,
        regularization: Any | None = ...,
        name: str | None = ...,
    ) -> None: ...
    def apply(self, executor: Any, need_restore: bool = ...) -> None: ...
    def restore(self, executor: Any) -> None: ...

class ExponentialMovingAverage:
    apply_program: Any = ...
    restore_program: Any = ...
    def __init__(self, decay: float = ..., thres_steps: Any | None = ..., name: str | None = ...) -> None: ...
    def update(self) -> None: ...
    def apply(self, executor: Any, need_restore: bool = ...) -> None: ...
    def restore(self, executor: Any) -> None: ...

class PipelineOptimizer:
    output_var_to_op: Any = ...
    input_var_to_op: Any = ...
    def __init__(self, optimizer: Any, num_microbatches: int = ..., start_cpu_core_id: int = ...) -> None: ...
    origin_main_block: Any = ...
    local_rank: Any = ...
    schedule_mode: Any = ...
    micro_batch_size: Any = ...
    use_sharding: Any = ...
    ring_id: Any = ...
    global_ring_id: Any = ...
    mp_degree: Any = ...
    mp_rank: Any = ...
    scale_gradient: Any = ...
    def minimize(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
    ): ...

class RecomputeOptimizer(Optimizer):
    enable_offload: bool = ...
    def __init__(self, optimizer: Any) -> None: ...
    def load(self, state_dict: Any) -> None: ...
    def apply_gradients(self, params_grads: Any): ...
    sorted_checkpoint_names: Any = ...
    def backward(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
        callbacks: Any | None = ...,
    ): ...
    def apply_optimize(self, loss: Any, startup_program: Any, params_grads: Any): ...
    def minimize(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
    ): ...

class LookaheadOptimizer:
    inner_optimizer: Any = ...
    alpha: Any = ...
    k: Any = ...
    type: str = ...
    def __init__(self, inner_optimizer: Any, alpha: float = ..., k: int = ...) -> None: ...
    def minimize(self, loss: Any, startup_program: Any | None = ...): ...

class GradientMergeOptimizer:
    GRAD_MERGE_COND_NAME: str = ...
    inner_optimizer: Any = ...
    k_steps: Any = ...
    type: str = ...
    avg: Any = ...
    def __init__(self, inner_optimizer: Any, k_steps: int = ..., avg: bool = ...) -> None: ...
    def backward(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
        callbacks: Any | None = ...,
    ): ...
    def apply_optimize(self, loss: Any, startup_program: Any, params_grads: Any): ...
    def apply_gradients(self, params_grads: Any): ...
    def minimize(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
    ): ...
