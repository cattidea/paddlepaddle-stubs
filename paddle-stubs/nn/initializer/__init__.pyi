from __future__ import annotations

from typing import Optional

from ...fluid.initializer import Bilinear as Bilinear
from ...fluid.initializer import Initializer as Initializer
from ...fluid.initializer import set_global_initializer as set_global_initializer
from .assign import Assign as Assign
from .constant import Constant as Constant
from .dirac import Dirac as Dirac
from .kaiming import KaimingNormal as KaimingNormal
from .kaiming import KaimingUniform as KaimingUniform
from .normal import Normal as Normal
from .normal import TruncatedNormal as TruncatedNormal
from .orthogonal import Orthogonal as Orthogonal
from .uniform import Uniform as Uniform
from .xavier import XavierNormal as XavierNormal
from .xavier import XavierUniform as XavierUniform

def calculate_gain(nonlinearity: str, param: Optional[int | float]) -> float: ...
