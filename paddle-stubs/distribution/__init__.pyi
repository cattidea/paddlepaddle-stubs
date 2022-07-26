from __future__ import annotations

from paddle.distribution.beta import Beta as Beta
from paddle.distribution.categorical import Categorical as Categorical
from paddle.distribution.dirichlet import Dirichlet as Dirichlet
from paddle.distribution.distribution import Distribution as Distribution
from paddle.distribution.exponential_family import (
    ExponentialFamily as ExponentialFamily,
)
from paddle.distribution.independent import Independent as Independent
from paddle.distribution.kl import kl_divergence as kl_divergence
from paddle.distribution.kl import register_kl as register_kl
from paddle.distribution.multinomial import Multinomial as Multinomial
from paddle.distribution.normal import Normal as Normal
from paddle.distribution.transform import *
from paddle.distribution.transformed_distribution import (
    TransformedDistribution as TransformedDistribution,
)
from paddle.distribution.uniform import Uniform as Uniform
