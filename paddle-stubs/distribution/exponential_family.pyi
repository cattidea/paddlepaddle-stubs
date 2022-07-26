from __future__ import annotations

from paddle.distribution import distribution
from paddle.fluid.framework import in_dygraph_mode as in_dygraph_mode

class ExponentialFamily(distribution.Distribution):
    def entropy(self): ...
