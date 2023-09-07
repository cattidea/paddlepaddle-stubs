from __future__ import annotations

from typing import Any

from .. import core as core
from .. import layers as layers

def default_collate_fn(batch: Any): ...
def default_convert_fn(batch: Any): ...
