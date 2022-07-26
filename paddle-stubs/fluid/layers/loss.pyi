from __future__ import annotations

from typing import Any, Optional

def center_loss(input: Any, label: Any, num_classes: Any, alpha: Any, param_attr: Any, update_center: bool = ...): ...
def bpr_loss(input: Any, label: Any, name: Optional[Any] = ...): ...
def cross_entropy(input: Any, label: Any, soft_label: bool = ..., ignore_index: Any = ...): ...
def square_error_cost(input: Any, label: Any): ...
def edit_distance(
    input: Any,
    label: Any,
    normalized: bool = ...,
    ignored_tokens: Optional[Any] = ...,
    input_length: Optional[Any] = ...,
    label_length: Optional[Any] = ...,
): ...
def warpctc(
    input: Any,
    label: Any,
    blank: int = ...,
    norm_by_times: bool = ...,
    input_length: Optional[Any] = ...,
    label_length: Optional[Any] = ...,
): ...
def nce(
    input: Any,
    label: Any,
    num_total_classes: Any,
    sample_weight: Optional[Any] = ...,
    param_attr: Optional[Any] = ...,
    bias_attr: Optional[Any] = ...,
    num_neg_samples: Optional[Any] = ...,
    name: Optional[Any] = ...,
    sampler: str = ...,
    custom_dist: Optional[Any] = ...,
    seed: int = ...,
    is_sparse: bool = ...,
): ...
def hsigmoid(
    input: Any,
    label: Any,
    num_classes: Any,
    param_attr: Optional[Any] = ...,
    bias_attr: Optional[Any] = ...,
    name: Optional[Any] = ...,
    path_table: Optional[Any] = ...,
    path_code: Optional[Any] = ...,
    is_custom: bool = ...,
    is_sparse: bool = ...,
): ...
def sampled_softmax_with_cross_entropy(
    logits: Any,
    label: Any,
    num_samples: Any,
    num_true: int = ...,
    remove_accidental_hits: bool = ...,
    use_customized_samples: bool = ...,
    customized_samples: Optional[Any] = ...,
    customized_probabilities: Optional[Any] = ...,
    seed: int = ...,
): ...
def softmax_with_cross_entropy(
    logits: Any,
    label: Any,
    soft_label: bool = ...,
    ignore_index: Any = ...,
    numeric_stable_mode: bool = ...,
    return_softmax: bool = ...,
    axis: int = ...,
): ...
def rank_loss(label: Any, left: Any, right: Any, name: Optional[Any] = ...): ...
def margin_rank_loss(label: Any, left: Any, right: Any, margin: float = ..., name: Optional[Any] = ...): ...
def sigmoid_cross_entropy_with_logits(
    x: Any, label: Any, ignore_index: Any = ..., name: Optional[Any] = ..., normalize: bool = ...
): ...
def teacher_student_sigmoid_loss(
    input: Any, label: Any, soft_max_up_bound: float = ..., soft_max_lower_bound: Any = ...
): ...
def huber_loss(input: Any, label: Any, delta: Any): ...
def kldiv_loss(x: Any, target: Any, reduction: str = ..., name: Optional[Any] = ...): ...
def npair_loss(anchor: Any, positive: Any, labels: Any, l2_reg: float = ...): ...
def mse_loss(input: Any, label: Any): ...
