from __future__ import annotations

from typing import Any

class WeightDecayHelper:
    def __init__(self) -> None: ...
    def prune_weight_decay(self, block: Any, shard: Any) -> None: ...
