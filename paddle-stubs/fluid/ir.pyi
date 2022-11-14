from __future__ import annotations

from typing import Any, Optional

import paddle

from . import core as core
from . import unique_name as unique_name
from .framework import OpProtoHolder as OpProtoHolder

def get_data_vars(program: Any): ...
def apply_build_strategy(main_program: Any, startup_program: Any, build_strategy: Any, pass_attrs: Any): ...

class RegisterPassHelper:
    def __init__(self, pass_pairs: Any, pass_type: Any = ..., input_specs: Any = ...) -> None: ...
    def SerializeMultiPassDesc(self): ...

class PassDesc:
    class AttrHelper:
        def __init__(self, obj: Any, name: Any, element_index: Any | None = ...) -> None: ...
        def __getitem__(self, index: Any): ...
        def __sub__(self, value: Any): ...
        def __add__(self, value: Any): ...
        def Mod(self, value: Any): ...
        def Size(self): ...
        def EQ(self, value: Any) -> None: ...
        def MappedPattern(
            self,
            var: Any | None = ...,
            op: Any | None = ...,
            index: int = ...,
            name: str | None = ...,
            element_index: Any | None = ...,
        ): ...

    class VarHelper(paddle.static.Variable):
        def __init__(self, *args: Any, **kwargs: Any) -> None: ...
        def __getattr__(self, name: Any): ...
        def Attr(self, name: Any): ...

    class OpHelper:
        def __init__(self, type: Any | None = ...) -> None: ...
        def __getattr__(self, name: Any): ...
        def __call__(self, *args: Any, **kwargs: Any): ...
        def Init(self) -> None: ...
        def Attr(self, name: Any): ...
        def SetAttr(self, name: Any, value: Any) -> None: ...
        def Output(self, name: Any): ...
        def Outputs(self): ...
        def SetOutputs(self, **kwargs: Any) -> None: ...
    OP: Any = ...

def RegisterPass(function: Any | None = ..., input_specs: Any = ...): ...
