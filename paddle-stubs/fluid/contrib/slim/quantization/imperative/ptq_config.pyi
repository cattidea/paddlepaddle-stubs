from __future__ import annotations

from typing import Any

from .ptq_quantizer import *

class PTQConfig:
    in_act_quantizer: Any = ...
    out_act_quantizer: Any = ...
    wt_quantizer: Any = ...
    quant_hook_handle: Any = ...
    enable_in_act_quantizer: bool = ...
    def __init__(self, activation_quantizer: Any, weight_quantizer: Any) -> None: ...

default_ptq_config: Any
