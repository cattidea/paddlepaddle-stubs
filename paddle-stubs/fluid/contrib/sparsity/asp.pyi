from __future__ import annotations

from typing import Any, Optional

def set_excluded_layers(main_program: Any, param_names: Any) -> None: ...
def reset_excluded_layers(main_program: Any | None = ...) -> None: ...
def decorate(optimizer: Any): ...
def prune_model(
    main_program: Any | None = ..., n: int = ..., m: int = ..., mask_algo: str = ..., with_mask: bool = ...
): ...

class ProgramASPInfo:
    def __init__(self) -> None: ...
    def update_mask_vars(self, param_name: Any, var: Any) -> None: ...
    def update_masks(self, param_name: Any, var: Any) -> None: ...
    def update_excluded_layers(self, param_names: Any) -> None: ...
    def reset_excluded_layers(self) -> None: ...
    @property
    def mask_vars(self): ...
    @property
    def masks(self): ...
    @property
    def excluded_layers(self): ...

class ASPHelper:
    MASK_APPENDDED_NAME: str = ...
    PADDLE_WEIGHT_SUFFIX: str = ...
    @classmethod
    def set_excluded_layers(cls, main_program: Any, param_names: Any) -> None: ...
    @classmethod
    def reset_excluded_layers(cls, main_program: Any | None = ...) -> None: ...
    @staticmethod
    def decorate(optimizer: Any): ...
    @classmethod
    def prune_model(
        cls,
        place: Any,
        main_program: Any | None = ...,
        n: int = ...,
        m: int = ...,
        mask_algo: Any = ...,
        with_mask: bool = ...,
    ): ...

class OptimizerWithSparsityGuarantee:
    def __init__(self, optimizer: Any) -> None: ...
    def minimize(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
    ): ...
