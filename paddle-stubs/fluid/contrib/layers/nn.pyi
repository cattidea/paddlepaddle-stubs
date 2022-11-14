from __future__ import annotations

from typing import Any, Optional

def fused_elemwise_activation(
    x: Any, y: Any, functor_list: Any, axis: int = ..., scale: float = ..., save_intermediate_out: bool = ...
): ...
def var_conv_2d(
    input: Any,
    row: Any,
    col: Any,
    input_channel: Any,
    output_channel: Any,
    filter_size: Any,
    stride: int = ...,
    param_attr: Any | None = ...,
    act: Any | None = ...,
    dtype: str = ...,
    name: str | None = ...,
): ...
def match_matrix_tensor(
    x: Any,
    y: Any,
    channel_num: Any,
    act: Any | None = ...,
    param_attr: Any | None = ...,
    dtype: str = ...,
    name: str | None = ...,
): ...
def sequence_topk_avg_pooling(input: Any, row: Any, col: Any, topks: Any, channel_num: Any): ...
def tree_conv(
    nodes_vector: Any,
    edge_set: Any,
    output_size: Any,
    num_filters: int = ...,
    max_depth: int = ...,
    act: str = ...,
    param_attr: Any | None = ...,
    bias_attr: Any | None = ...,
    name: str | None = ...,
): ...
def fused_embedding_seq_pool(
    input: Any,
    size: Any,
    is_sparse: bool = ...,
    padding_idx: Any | None = ...,
    combiner: str = ...,
    param_attr: Any | None = ...,
    dtype: str = ...,
): ...
def fused_seqpool_cvm(
    input: Any, pool_type: Any, cvm: Any, pad_value: float = ..., use_cvm: bool = ..., cvm_offset: int = ...
): ...
def multiclass_nms2(
    bboxes: Any,
    scores: Any,
    score_threshold: Any,
    nms_top_k: Any,
    keep_top_k: Any,
    nms_threshold: float = ...,
    normalized: bool = ...,
    nms_eta: float = ...,
    background_label: int = ...,
    return_index: bool = ...,
    name: str | None = ...,
): ...
def search_pyramid_hash(
    input: Any,
    num_emb: Any,
    space_len: Any,
    pyramid_layer: Any,
    rand_len: Any,
    drop_out_percent: Any,
    is_training: Any,
    use_filter: Any,
    white_list_len: Any,
    black_list_len: Any,
    seed: Any,
    lr: Any,
    param_attr: Any | None = ...,
    param_attr_wl: Any | None = ...,
    param_attr_bl: Any | None = ...,
    name: str | None = ...,
    distribute_update_vars: Any | None = ...,
    dtype: str = ...,
): ...
def shuffle_batch(x: Any, seed: Any | None = ...): ...
def partial_concat(input: Any, start_index: int = ..., length: int = ...): ...
def partial_sum(input: Any, start_index: int = ..., length: int = ...): ...
def sparse_embedding(
    input: Any,
    size: Any,
    padding_idx: Any | None = ...,
    is_test: bool = ...,
    entry: Any | None = ...,
    table_class: str = ...,
    param_attr: Any | None = ...,
    dtype: str = ...,
): ...
def tdm_child(x: Any, node_nums: Any, child_nums: Any, param_attr: Any | None = ..., dtype: str = ...): ...
def tdm_sampler(
    x: Any,
    neg_samples_num_list: Any,
    layer_node_num_list: Any,
    leaf_node_num: Any,
    tree_travel_attr: Any | None = ...,
    tree_layer_attr: Any | None = ...,
    output_positive: bool = ...,
    output_list: bool = ...,
    seed: int = ...,
    tree_dtype: str = ...,
    dtype: str = ...,
): ...
def rank_attention(
    input: Any, rank_offset: Any, rank_param_shape: Any, rank_param_attr: Any, max_rank: int = ..., max_size: int = ...
): ...
def batch_fc(input: Any, param_size: Any, param_attr: Any, bias_size: Any, bias_attr: Any, act: Any | None = ...): ...
def bilateral_slice(x: Any, guide: Any, grid: Any, has_offset: Any, name: str | None = ...): ...
def correlation(
    x: Any,
    y: Any,
    pad_size: Any,
    kernel_size: Any,
    max_displacement: Any,
    stride1: Any,
    stride2: Any,
    corr_type_multiply: int = ...,
): ...
def fused_bn_add_act(
    x: Any,
    y: Any,
    momentum: float = ...,
    epsilon: float = ...,
    param_attr: Any | None = ...,
    bias_attr: Any | None = ...,
    moving_mean_name: str | None = ...,
    moving_variance_name: str | None = ...,
    act: Any | None = ...,
    name: str | None = ...,
): ...

# Names in __all__ with no definition:
#   _pull_box_extended_sparse
