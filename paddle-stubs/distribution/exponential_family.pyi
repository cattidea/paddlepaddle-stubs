from __future__ import annotations

from paddle.base.framework import in_dygraph_mode as in_dygraph_mode
from paddle.distribution import distribution

class ExponentialFamily(distribution.Distribution):
    def entropy(self): ...
