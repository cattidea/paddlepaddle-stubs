from __future__ import annotations

from paddle.base import core as core
from paddle.base import dygraph_utils as dygraph_utils
from paddle.base.core import VarDesc as VarDesc
from paddle.base.data_feeder import check_dtype as check_dtype
from paddle.base.data_feeder import check_type as check_type
from paddle.base.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from paddle.base.data_feeder import convert_dtype as convert_dtype
from paddle.base.framework import OpProtoHolder as OpProtoHolder
from paddle.base.framework import Variable as Variable
from paddle.base.framework import (
    convert_np_dtype_to_dtype_ as convert_np_dtype_to_dtype_,
)
from paddle.base.framework import default_main_program as default_main_program
from paddle.base.framework import device_guard as device_guard
from paddle.base.framework import dygraph_only as dygraph_only
from paddle.base.initializer import Constant as Constant
from paddle.base.layer_helper import LayerHelper as LayerHelper
from paddle.base.layers import fill_constant as fill_constant
from paddle.base.layers import scale as scale
from paddle.base.layers import utils as utils
from paddle.base.layers.layer_function_generator import templatedoc as templatedoc
from paddle.base.param_attr import ParamAttr as ParamAttr
from six.moves import reduce as reduce
