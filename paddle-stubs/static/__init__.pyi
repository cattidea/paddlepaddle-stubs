from __future__ import annotations

from typing import Any

from typing_extensions import TypeAlias

BuildStrategy: TypeAlias = Any

from ..base.backward import append_backward as append_backward
from ..base.backward import gradients as gradients
from ..base.compiler import CompiledProgram as CompiledProgram
from ..base.compiler import ExecutionStrategy as ExecutionStrategy
from ..base.compiler import IpuCompiledProgram as IpuCompiledProgram
from ..base.compiler import IpuStrategy as IpuStrategy
from ..base.executor import Executor as Executor
from ..base.executor import global_scope as global_scope
from ..base.executor import scope_guard as scope_guard
from ..base.framework import Program as Program
from ..base.framework import Variable as Variable
from ..base.framework import cpu_places as cpu_places
from ..base.framework import cuda_places as cuda_places
from ..base.framework import default_main_program as default_main_program
from ..base.framework import default_startup_program as default_startup_program
from ..base.framework import device_guard as device_guard
from ..base.framework import ipu_shard_guard as ipu_shard_guard
from ..base.framework import mlu_places as mlu_places
from ..base.framework import name_scope as name_scope
from ..base.framework import npu_places as npu_places
from ..base.framework import program_guard as program_guard
from ..base.framework import xpu_places as xpu_places
from ..base.io import load as load
from ..base.io import load_program_state as load_program_state
from ..base.io import save as save
from ..base.io import set_program_state as set_program_state
from ..base.layers import create_global_var as create_global_var
from ..base.layers import create_parameter as create_parameter
from ..base.layers.control_flow import Print as Print
from ..base.layers.metric_op import accuracy as accuracy
from ..base.layers.metric_op import auc as auc
from ..base.layers.nn import py_func as py_func
from ..base.optimizer import ExponentialMovingAverage as ExponentialMovingAverage
from ..base.parallel_executor import ParallelExecutor as ParallelExecutor
from ..base.param_attr import WeightNormParamAttr as WeightNormParamAttr
from .input import InputSpec as InputSpec
from .input import data as data
from .io import deserialize_persistables as deserialize_persistables
from .io import deserialize_program as deserialize_program
from .io import load_from_file as load_from_file
from .io import load_inference_model as load_inference_model
from .io import normalize_program as normalize_program
from .io import save_inference_model as save_inference_model
from .io import save_to_file as save_to_file
from .io import serialize_persistables as serialize_persistables
from .io import serialize_program as serialize_program
