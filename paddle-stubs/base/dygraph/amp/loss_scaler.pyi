from __future__ import annotations

from enum import Enum
from typing import Any

class OptimizerState(Enum):
    INIT = ...
    UNSCALED = ...
    STEPPED = ...

class AmpScaler:
    def __init__(
        self,
        enable: bool = ...,
        init_loss_scaling: Any = ...,
        incr_ratio: float = ...,
        decr_ratio: float = ...,
        incr_every_n_steps: int = ...,
        decr_every_n_nan_or_inf: int = ...,
        use_dynamic_loss_scaling: bool = ...,
    ) -> None: ...
    def scale(self, var: Any): ...
    def minimize(self, optimizer: Any, *args: Any, **kwargs: Any): ...
    def is_enable(self): ...
    def is_use_dynamic_loss_scaling(self): ...
    def get_init_loss_scaling(self): ...
    def set_init_loss_scaling(self, new_init_loss_scaling: Any) -> None: ...
    def get_incr_ratio(self): ...
    def set_incr_ratio(self, new_incr_ratio: Any) -> None: ...
    def get_decr_ratio(self): ...
    def set_decr_ratio(self, new_decr_ratio: Any) -> None: ...
    def get_incr_every_n_steps(self): ...
    def set_incr_every_n_steps(self, new_incr_every_n_steps: Any) -> None: ...
    def get_decr_every_n_nan_or_inf(self): ...
    def set_decr_every_n_nan_or_inf(self, new_decr_every_n_nan_or_inf: Any) -> None: ...
    def state_dict(self): ...
    def load_state_dict(self, state_dict: Any) -> None: ...