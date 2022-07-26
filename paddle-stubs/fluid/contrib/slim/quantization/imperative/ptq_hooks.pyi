from __future__ import annotations

from typing import Any

from . import ptq_config as ptq_config
from .ptq_registry import PTQRegistry as PTQRegistry

def quant_forward_post_hook(layer: Any, inputs: Any, outputs: Any) -> None: ...
