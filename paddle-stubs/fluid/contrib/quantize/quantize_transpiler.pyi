from __future__ import annotations

from typing import Any, Optional

class QuantizeTranspiler:
    weight_bits: Any = ...
    activation_bits: Any = ...
    weight_quantize_type: Any = ...
    activation_quantize_type: Any = ...
    window_size: Any = ...
    moving_rate: Any = ...
    helper: Any = ...
    fake_quant_op_types: Any = ...
    fake_dequant_op_types: Any = ...
    is_test: Any = ...
    global_step: Any = ...
    def __init__(
        self,
        weight_bits: int = ...,
        activation_bits: int = ...,
        activation_quantize_type: str = ...,
        weight_quantize_type: str = ...,
        window_size: int = ...,
        moving_rate: float = ...,
    ) -> None: ...
    def training_transpile(self, program: Any | None = ..., startup_program: Any | None = ...) -> None: ...
    def freeze_program(self, program: Any, place: Any, scope: Any | None = ...): ...
    def convert_to_int8(self, program: Any, place: Any, scope: Any | None = ...): ...
