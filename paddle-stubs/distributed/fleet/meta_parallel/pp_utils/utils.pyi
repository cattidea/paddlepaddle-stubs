from __future__ import annotations

from typing import Any

from paddle.autograd import EagerPyLayer, PyLayer

from ...utils.recompute import check_recompute_necessary as check_recompute_necessary
from ...utils.recompute import detach_variable as detach_variable
from ..parallel_layers.random import get_rng_state_tracker as get_rng_state_tracker

FLOAT_TYPE_DICT: Any
PADDLE_TO_NUMBER: Any
NUMBER_TO_DTYPE: Any

def is_float_tensor(tensor: Any): ...
def get_tensor_dtype(dtype: Any): ...
def paddle_2_number(dtype: Any): ...
def number_2_dtype(number: Any): ...
def get_tensor_bytes(tensor: Any): ...

class _HPEagerRecomputeFunction(EagerPyLayer):
    @staticmethod
    def forward(ctx: Any, run_function: Any, all_outputs: Any, *args: Any): ...
    @staticmethod
    def backward(ctx: Any, *args: Any): ...

class _HPRecomputeFunction(PyLayer):
    @staticmethod
    def forward(ctx: Any, run_function: Any, all_outputs: Any, *args: Any): ...
    @staticmethod
    def backward(ctx: Any, *args: Any): ...
