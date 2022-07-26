from __future__ import annotations

from typing import Any

from paddle.autograd import EagerPyLayer, PyLayer

logger: Any
formatter: Any
ch: Any

def detach_variable(inputs: Any): ...
def check_recompute_necessary(inputs: Any) -> None: ...
def swith_rng_state(rng_state: Any) -> None: ...

class EagerRecomputeFunction(EagerPyLayer):
    @staticmethod
    def forward(ctx: Any, run_function: Any, preserve_rng_state: Any, *args: Any): ...
    @staticmethod
    def backward(ctx: Any, *args: Any): ...

class RecomputeFunction(PyLayer):
    @staticmethod
    def forward(ctx: Any, run_function: Any, preserve_rng_state: Any, *args: Any): ...
    @staticmethod
    def backward(ctx: Any, *args: Any): ...

def recompute(function: Any, *args: Any, **kwargs: Any): ...
