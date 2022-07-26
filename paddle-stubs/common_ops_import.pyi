from __future__ import annotations

from paddle.fluid import core as core
from paddle.fluid import dygraph_utils as dygraph_utils
from paddle.fluid.core import VarDesc as VarDesc
from paddle.fluid.data_feeder import check_dtype as check_dtype
from paddle.fluid.data_feeder import check_type as check_type
from paddle.fluid.data_feeder import (
    check_variable_and_dtype as check_variable_and_dtype,
)
from paddle.fluid.data_feeder import convert_dtype as convert_dtype
from paddle.fluid.framework import OpProtoHolder as OpProtoHolder
from paddle.fluid.framework import Variable as Variable
from paddle.fluid.framework import (
    convert_np_dtype_to_dtype_ as convert_np_dtype_to_dtype_,
)
from paddle.fluid.framework import default_main_program as default_main_program
from paddle.fluid.framework import device_guard as device_guard
from paddle.fluid.framework import dygraph_only as dygraph_only
from paddle.fluid.initializer import Constant as Constant
from paddle.fluid.layer_helper import LayerHelper as LayerHelper
from paddle.fluid.layers import fill_constant as fill_constant
from paddle.fluid.layers import scale as scale
from paddle.fluid.layers import utils as utils
from paddle.fluid.layers.layer_function_generator import templatedoc as templatedoc
from paddle.fluid.param_attr import ParamAttr as ParamAttr
from six.moves import reduce as reduce
