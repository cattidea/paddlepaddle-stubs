from __future__ import annotations

from typing import Any

from .cluster import DeviceType as DeviceType
from .graph import Edge as Edge
from .graph import Graph as Graph
from .graph import Node as Node
from .process_group import get_process_group as get_process_group

def is_collective_comm_op(op: Any): ...
def is_p2p_comm_op(op: Any): ...
def get_dtype_bytes(dtype: Any): ...
def get_comm_volume(comm_op: Any, src_rank: Any, tgt_rank: Any): ...
def analyze_comm_requirements_from_op(op: Any, rank: Any, g_process_group_map: Any): ...
def analyze_requirements_for_program(src_info: Any, rank: Any): ...
def build_process_graph(distributed_program: Any): ...
def build_cluster_graph(cluster: Any): ...
def mapping(distributed_program: Any, cluster: Any): ...
