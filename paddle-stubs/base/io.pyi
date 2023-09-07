from __future__ import annotations

from functools import reduce as reduce
from typing import Any, Optional

from paddle.base import layers as layers
from paddle.base.evaluator import Evaluator as Evaluator
from paddle.base.framework import default_startup_program as default_startup_program
from paddle.reader import ComposeNotAligned as ComposeNotAligned
from paddle.reader import buffered as buffered
from paddle.reader import cache as cache
from paddle.reader import chain as chain
from paddle.reader import compose as compose
from paddle.reader import firstn as firstn
from paddle.reader import map_readers as map_readers
from paddle.reader import multiprocess_reader as multiprocess_reader
from paddle.reader import shuffle as shuffle
from paddle.reader import xmap_readers as xmap_readers

from . import core as core
from . import dataloader as dataloader
from . import reader as reader
from . import unique_name as unique_name
from .dataloader import *
from .reader import *
from .wrapped_decorator import (
    signature_safe_contextmanager as signature_safe_contextmanager,
)

batch: Any

class _open_buffer:
    buffer: Any = ...
    def __init__(self, buffer: Any) -> None: ...
    def __enter__(self): ...

class _buffer_reader(_open_buffer):
    initial_tell: Any = ...
    def __init__(self, buffer: Any) -> None: ...
    def __exit__(self, *args: Any) -> None: ...

class _buffer_writer(_open_buffer):
    def __exit__(self, *args: Any) -> None: ...

def is_parameter(var: Any): ...
def is_persistable(var: Any): ...
def is_belong_to_optimizer(var: Any): ...
def get_program_parameter(program: Any): ...
def get_program_persistable_vars(program: Any): ...
def save_vars(
    executor: Any,
    dirname: Any,
    main_program: Any | None = ...,
    vars: Any | None = ...,
    predicate: Any | None = ...,
    filename: str | None = ...,
): ...
def save_params(executor: Any, dirname: Any, main_program: Any | None = ..., filename: str | None = ...): ...
def save_persistables(executor: Any, dirname: Any, main_program: Any | None = ..., filename: str | None = ...): ...
def load_vars(
    executor: Any,
    dirname: Any,
    main_program: Any | None = ...,
    vars: Any | None = ...,
    predicate: Any | None = ...,
    filename: str | None = ...,
) -> None: ...
def load_params(executor: Any, dirname: Any, main_program: Any | None = ..., filename: str | None = ...) -> None: ...
def load_persistables(
    executor: Any, dirname: Any, main_program: Any | None = ..., filename: str | None = ...
) -> None: ...
def prepend_feed_ops(inference_program: Any, feed_target_names: Any, feed_holder_name: str = ...) -> None: ...
def append_fetch_ops(inference_program: Any, fetch_target_names: Any, fetch_holder_name: str = ...) -> None: ...
def save_inference_model(
    dirname: Any,
    feeded_var_names: Any,
    target_vars: Any,
    executor: Any,
    main_program: Any | None = ...,
    model_filename: str | None = ...,
    params_filename: str | None = ...,
    export_for_deployment: bool = ...,
    program_only: bool = ...,
    clip_extra: bool = ...,
): ...
def load_inference_model(
    dirname: Any,
    executor: Any,
    model_filename: str | None = ...,
    params_filename: str | None = ...,
    pserver_endpoints: Any | None = ...,
): ...
def get_parameter_value(para: Any, executor: Any): ...
def get_parameter_value_by_name(name: Any, executor: Any, program: Any | None = ...): ...
def save(program: Any, model_path: Any, protocol: int = ..., **configs: Any): ...
def load(program: Any, model_path: Any, executor: Any | None = ..., var_list: Any | None = ...) -> None: ...
def load_program_state(model_path: Any, var_list: Any | None = ...): ...
def set_program_state(program: Any, state_dict: Any) -> None: ...
