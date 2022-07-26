from __future__ import annotations

from typing import Any, Optional

from ..framework import unique_name as unique_name
from ..layer_helper import LayerHelper as LayerHelper
from .layer_function_generator import autodoc as autodoc

def get_places(device_count: Optional[Any] = ..., device_type: Optional[Any] = ...): ...
