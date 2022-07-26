from __future__ import annotations

from typing import Any

from paddle.distributed.fleet.launch_utils import DistributeMode as DistributeMode

from .collective import CollectiveLauncher as CollectiveLauncher
from .manager import ELASTIC_EXIT_CODE as ELASTIC_EXIT_CODE
from .manager import ElasticLevel as ElasticLevel
from .manager import ElasticManager as ElasticManager
from .manager import ElasticStatus as ElasticStatus

def enable_elastic(args: Any, distribute_mode: Any): ...
def launch_elastic(args: Any, distribute_mode: Any) -> None: ...
