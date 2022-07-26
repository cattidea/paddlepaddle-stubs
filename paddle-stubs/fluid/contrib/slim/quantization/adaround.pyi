from __future__ import annotations

from typing import Any, Optional

from ....log_helper import get_logger as get_logger
from .utils import calculate_quant_cos_error as calculate_quant_cos_error
from .utils import dequant_tensor as dequant_tensor
from .utils import load_variable_data as load_variable_data
from .utils import quant_tensor as quant_tensor
from .utils import set_variable_data as set_variable_data
from .utils import stable_sigmoid as stable_sigmoid

GAMMA: Any
ZETA: float

def compute_soft_rounding(alpha_v: Any): ...
def compute_soft_rounding_np(alpha_v: Any): ...

class AdaRoundLoss:
    default_reg_param: Any = ...
    default_beta_range: Any = ...
    def __init__(self, reg_param: float = ..., default_beta_range: Any = ...) -> None: ...
    def compute_recon_loss(self, ada_quantized_output: Any, orig_output: Any): ...
    def compute_round_loss(self, alpha_v: Any, warm_start: Any, beta: Any): ...
    def compute_beta(self, max_iter: Any, cur_iter: Any, warm_start: Any): ...

class AdaRound:
    is_train: Any = ...
    num_iterations: Any = ...
    warm_start: float = ...
    weight_bits: int = ...
    offset: float = ...
    adaround_loss: Any = ...
    ori_weight_tensor: Any = ...
    scale: Any = ...
    scope: Any = ...
    quant_axis: int = ...
    weight_var_name: Any = ...
    alpha_name: Any = ...
    def __init__(
        self,
        scale: Any,
        weight_tensor: Any,
        scope: Optional[Any] = ...,
        weight_var_name: Optional[Any] = ...,
        weight_op_type: Optional[Any] = ...,
        is_train: bool = ...,
        num_iterations: int = ...,
    ) -> None: ...
    alpha_v: Any = ...
    def initialize_alpha(self, tensor: Any, scale: Any, var_name: Any) -> None: ...
    def update_final_weights(self): ...
    def get_loss(self, beta: Any, warm_start: Any, adaround_out_tensor: Any, orig_out_tensor: Any): ...
    def update_beta_warm(self, cur_iteration: Any): ...

def run_adaround(
    data_loader: Any,
    fp32_program: Any,
    fetch_list: Any,
    exe: Any,
    scope: Any,
    place: Any,
    quantized_op_pairs: Any,
    weight_op_pairs: Any,
    scale_dict: Any,
    num_iterations: int = ...,
    lr: float = ...,
    fast_mode: bool = ...,
) -> None: ...
