from __future__ import annotations

from typing import Any

from . import average as average
from . import backward as backward
from . import clip as clip
from . import compiler as compiler
from . import contrib as contrib
from . import core as core
from . import data_feed_desc as data_feed_desc
from . import dataset as dataset
from . import distribute_lookup_table as distribute_lookup_table
from . import dygraph as dygraph
from . import evaluator as evaluator
from . import executor as executor
from . import framework as framework
from . import generator as generator
from . import incubate as incubate
from . import initializer as initializer
from . import install_check as install_check
from . import io as io
from . import layers as layers
from . import metrics as metrics
from . import nets as nets
from . import optimizer as optimizer
from . import parallel_executor as parallel_executor
from . import profiler as profiler
from . import regularizer as regularizer
from . import trainer_desc as trainer_desc
from . import transpiler as transpiler
from . import unique_name as unique_name
from .backward import append_backward as append_backward
from .backward import gradients as gradients
from .compiler import *
from .data import *
from .data_feed_desc import *
from .data_feeder import DataFeeder as DataFeeder
from .dataset import *
from .dygraph.base import disable_dygraph as disable_dygraph
from .dygraph.base import enable_dygraph as enable_dygraph
from .dygraph.checkpoint import load_dygraph as load_dygraph
from .dygraph.checkpoint import save_dygraph as save_dygraph
from .dygraph.layers import *
from .dygraph.nn import *
from .dygraph.varbase_patch_methods import monkey_patch_varbase as monkey_patch_varbase
from .executor import *
from .framework import *
from .generator import Generator as Generator
from .incubate import fleet as fleet
from .initializer import set_global_initializer as set_global_initializer
from .input import embedding as embedding
from .input import one_hot as one_hot
from .io import load as load
from .io import load_program_state as load_program_state
from .io import save as save
from .io import set_program_state as set_program_state
from .lod_tensor import create_lod_tensor as create_lod_tensor
from .lod_tensor import create_random_int_lodtensor as create_random_int_lodtensor
from .parallel_executor import *
from .param_attr import ParamAttr as ParamAttr
from .param_attr import WeightNormParamAttr as WeightNormParamAttr
from .trainer_desc import DistMultiTrainer as DistMultiTrainer
from .trainer_desc import HeterPipelineTrainer as HeterPipelineTrainer
from .trainer_desc import HeterXpuTrainer as HeterXpuTrainer
from .trainer_desc import MultiTrainer as MultiTrainer
from .trainer_desc import PipelineTrainer as PipelineTrainer
from .trainer_desc import TrainerDesc as TrainerDesc
from .transpiler import DistributeTranspiler as DistributeTranspiler
from .transpiler import DistributeTranspilerConfig as DistributeTranspilerConfig
from .transpiler import HashName as HashName
from .transpiler import RoundRobin as RoundRobin
from .transpiler import memory_optimize as memory_optimize
from .transpiler import release_memory as release_memory

core_suffix: str
legacy_core: Any
enable_imperative = enable_dygraph
disable_imperative = disable_dygraph
Tensor = Any

def __bootstrap__() -> None: ...
