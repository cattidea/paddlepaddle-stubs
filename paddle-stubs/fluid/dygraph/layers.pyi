from __future__ import annotations

from typing import Any, Optional

class HookRemoveHelper:
    next_hook_id: int = ...
    def __init__(self, hooks: Any) -> None: ...
    def remove(self) -> None: ...

class Layer:
    training: bool = ...
    def __init__(self, name_scope: Optional[Any] = ..., dtype: str = ...) -> None: ...
    def train(self) -> None: ...
    def eval(self) -> None: ...
    def apply(self, fn: Any): ...
    def full_name(self): ...
    def register_forward_post_hook(self, hook: Any): ...
    def register_forward_pre_hook(self, hook: Any): ...
    def create_parameter(
        self,
        shape: Any,
        attr: Optional[Any] = ...,
        dtype: Optional[Any] = ...,
        is_bias: bool = ...,
        default_initializer: Optional[Any] = ...,
    ): ...
    def create_variable(
        self, name: Optional[Any] = ..., persistable: Optional[Any] = ..., dtype: Optional[Any] = ...
    ): ...
    def create_tensor(
        self, name: Optional[Any] = ..., persistable: Optional[Any] = ..., dtype: Optional[Any] = ...
    ): ...
    def parameters(self, include_sublayers: bool = ...): ...
    def children(self) -> None: ...
    def named_children(self) -> None: ...
    def sublayers(self, include_self: bool = ...): ...
    def named_parameters(self, prefix: str = ..., include_sublayers: bool = ...) -> None: ...
    def named_sublayers(self, prefix: str = ..., include_self: bool = ..., layers_set: Optional[Any] = ...) -> None: ...
    def register_buffer(self, name: Any, tensor: Any, persistable: bool = ...) -> None: ...
    def buffers(self, include_sublayers: bool = ...): ...
    def named_buffers(self, prefix: str = ..., include_sublayers: bool = ...) -> None: ...
    def clear_gradients(self) -> None: ...
    def __call__(self, *inputs: Any, **kwargs: Any): ...
    def forward(self, *inputs: Any, **kwargs: Any) -> None: ...
    def backward(self, *inputs: Any) -> None: ...
    def add_sublayer(self, name: Any, sublayer: Any): ...
    def add_parameter(self, name: Any, parameter: Any): ...
    def __getattr__(self, name: Any): ...
    def __setattr__(self, name: Any, value: Any) -> None: ...
    def __delattr__(self, name: Any) -> None: ...
    def __dir__(self): ...
    def extra_repr(self): ...
    def register_state_dict_hook(self, hook: Any): ...
    def to_static_state_dict(
        self,
        destination: Optional[Any] = ...,
        include_sublayers: bool = ...,
        structured_name_prefix: str = ...,
        use_hook: bool = ...,
    ): ...
    def state_dict(
        self,
        destination: Optional[Any] = ...,
        include_sublayers: bool = ...,
        structured_name_prefix: str = ...,
        use_hook: bool = ...,
    ): ...
    def set_state_dict(self, state_dict: Any, use_structured_name: bool = ...): ...
    def to(self, device: Optional[Any] = ..., dtype: Optional[Any] = ..., blocking: Optional[Any] = ...): ...
    set_dict: Any = ...
    load_dict: Any = ...
