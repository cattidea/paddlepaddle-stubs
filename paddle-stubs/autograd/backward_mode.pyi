from __future__ import annotations

from typing import Any, Optional

from paddle.fluid.backward import gradients_with_optimizer as gradients_with_optimizer

def backward(tensors: Any, grad_tensors: Optional[Any] = ..., retain_graph: bool = ...): ...
