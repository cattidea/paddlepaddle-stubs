from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from typing import Any, Callable, Optional

from paddle import Tensor
from typing_extensions import ParamSpec, Self, TypeVar

from ..._typing import DTypeLike, PlaceLike, ShapeLike
from ...framework import ParamAttr
from ...nn.initializer import Initializer

HookParam = ParamSpec("HookParam")
Ret = TypeVar("Ret")

Hook = Callable[HookParam, Ret]  # [Generic], wait for PEP 695
PreHook = Hook[[Layer, Tensor], Tensor]  # (layer, input) -> transformed_input
PostHook = Hook[[Layer, Tensor, Tensor], Tensor]  # (layer, input, output) -> transformed_output
StateDict = dict[str, Tensor] | OrderedDict[str, Tensor]
StateDictHook = Hook[[StateDict], None]

class HookRemoveHelper:
    next_hook_id: int = ...
    def __init__(self, hooks: Any) -> None: ...
    def remove(self) -> None: ...

class Layer:
    training: bool = ...
    def __init__(self, name_scope: str | None = ..., dtype: DTypeLike = ...) -> None: ...

    # Children
    def children(self) -> Iterator[Self]: ...
    def named_children(self) -> Iterator[tuple[str, Self]]: ...
    # Sublayers
    def add_sublayer(self, name: str, sublayer: Self) -> Self: ...
    def sublayers(self, include_self: bool = ...) -> Iterator[Self]: ...
    def named_sublayers(
        self,
        prefix: str = ...,
        include_self: bool = ...,
        layers_set: set[Self] | None = ...,
    ) -> Iterator[tuple[str, Self]]: ...
    # Parameters
    def create_parameter(
        self,
        shape: ShapeLike,
        attr: ParamAttr | None = ...,
        dtype: DTypeLike | None = ...,
        is_bias: bool = ...,
        default_initializer: Initializer | None = ...,
    ) -> Tensor: ...
    def parameters(self, include_sublayers: bool = ...) -> list[Tensor]: ...
    def add_parameter(self, name: str, parameter: Tensor) -> Tensor: ...
    def named_parameters(
        self,
        prefix: str = ...,
        include_sublayers: bool = ...,
    ) -> Iterator[tuple[str, Tensor]]: ...
    # Tensor
    def create_tensor(
        self,
        name: str | None = ...,
        persistable: bool | None = ...,
        dtype: DTypeLike | None = ...,
    ) -> Tensor: ...
    # Buffer
    def register_buffer(self, name: str, tensor: Tensor, persistable: bool = ...) -> None: ...
    def buffers(self, include_sublayers: bool = ...) -> list[Tensor]: ...
    def named_buffers(
        self,
        prefix: str = ...,
        include_sublayers: bool = ...,
    ) -> Iterator[tuple[str, Tensor]]: ...

    # Training process
    def train(self) -> None: ...
    def eval(self) -> None: ...
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def backward(self, *inputs: Any) -> None: ...
    def clear_gradients(self) -> None: ...
    __call__ = forward

    # Hooks
    def register_forward_post_hook(self, hook: PostHook) -> HookRemoveHelper: ...
    def register_forward_pre_hook(self, hook: PreHook) -> HookRemoveHelper: ...

    # Magic methods related
    def __getattr__(self, name: str) -> Any: ...
    def __setattr__(self, name: str, value: Any) -> None: ...
    def __delattr__(self, name: str) -> None: ...
    def __dir__(self) -> list[str]: ...
    def extra_repr(self) -> str: ...

    # State dict
    def to_static_state_dict(
        self,
        destination: StateDict | None = ...,
        include_sublayers: bool = ...,
        structured_name_prefix: str = ...,
        use_hook: bool = ...,
    ) -> StateDict: ...
    def state_dict(
        self,
        destination: StateDict | None = ...,
        include_sublayers: bool = ...,
        structured_name_prefix: str = ...,
        use_hook: bool = ...,
    ) -> StateDict: ...
    def set_state_dict(
        self,
        state_dict: StateDict,
        use_structured_name: bool = ...,
    ) -> None: ...
    def register_state_dict_hook(self, hook: StateDictHook) -> HookRemoveHelper: ...
    # [aliases] Compatible with old method names
    set_dict = set_state_dict
    load_dict = set_state_dict

    # Others
    def to(
        self,
        device: PlaceLike | None = ...,
        dtype: DTypeLike | None = ...,
        blocking: bool | None = ...,
    ) -> Self: ...
    def apply(self, fn: Callable[[Self], None]) -> Self: ...
    def full_name(self) -> str: ...
