from __future__ import annotations

from typing import Any, Optional

from ...fluid.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ...fluid.framework import in_dygraph_mode as in_dygraph_mode
from ...fluid.layer_helper import LayerHelper as LayerHelper
from ...fluid.layers import dice_loss as dice_loss
from ...fluid.layers import edit_distance as edit_distance
from ...fluid.layers import huber_loss as huber_loss
from ...fluid.layers import log_loss as log_loss
from ...fluid.layers import npair_loss as npair_loss
from ...fluid.layers import square_error_cost as square_error_cost
from ...static import Variable as Variable
from ...tensor.manipulation import reshape as reshape

def binary_cross_entropy(
    input: Any, label: Any, weight: Optional[Any] = ..., reduction: str = ..., name: Optional[Any] = ...
): ...
def binary_cross_entropy_with_logits(
    logit: Any,
    label: Any,
    weight: Optional[Any] = ...,
    reduction: str = ...,
    pos_weight: Optional[Any] = ...,
    name: Optional[Any] = ...,
): ...
def hsigmoid_loss(
    input: Any,
    label: Any,
    num_classes: Any,
    weight: Any,
    bias: Optional[Any] = ...,
    path_table: Optional[Any] = ...,
    path_code: Optional[Any] = ...,
    is_sparse: bool = ...,
    name: Optional[Any] = ...,
): ...
def smooth_l1_loss(input: Any, label: Any, reduction: str = ..., delta: float = ..., name: Optional[Any] = ...): ...
def margin_ranking_loss(
    input: Any, other: Any, label: Any, margin: float = ..., reduction: str = ..., name: Optional[Any] = ...
): ...
def l1_loss(input: Any, label: Any, reduction: str = ..., name: Optional[Any] = ...): ...
def nll_loss(
    input: Any,
    label: Any,
    weight: Optional[Any] = ...,
    ignore_index: int = ...,
    reduction: str = ...,
    name: Optional[Any] = ...,
): ...
def kl_div(input: Any, label: Any, reduction: str = ..., name: Optional[Any] = ...): ...
def mse_loss(input: Any, label: Any, reduction: str = ..., name: Optional[Any] = ...): ...
def ctc_loss(
    log_probs: Any,
    labels: Any,
    input_lengths: Any,
    label_lengths: Any,
    blank: int = ...,
    reduction: str = ...,
    norm_by_times: bool = ...,
): ...
def margin_cross_entropy(
    logits: Any,
    label: Any,
    margin1: float = ...,
    margin2: float = ...,
    margin3: float = ...,
    scale: float = ...,
    group: Optional[Any] = ...,
    return_softmax: bool = ...,
    reduction: str = ...,
): ...
def softmax_with_cross_entropy(
    logits: Any,
    label: Any,
    soft_label: bool = ...,
    ignore_index: int = ...,
    numeric_stable_mode: bool = ...,
    return_softmax: bool = ...,
    axis: int = ...,
): ...
def cross_entropy(
    input: Any,
    label: Any,
    weight: Optional[Any] = ...,
    ignore_index: int = ...,
    reduction: str = ...,
    soft_label: bool = ...,
    axis: int = ...,
    use_softmax: bool = ...,
    name: Optional[Any] = ...,
): ...
def sigmoid_focal_loss(
    logit: Any,
    label: Any,
    normalizer: Optional[Any] = ...,
    alpha: float = ...,
    gamma: float = ...,
    reduction: str = ...,
    name: Optional[Any] = ...,
): ...
def hinge_embedding_loss(
    input: Any, label: Any, margin: float = ..., reduction: str = ..., name: Optional[Any] = ...
): ...
