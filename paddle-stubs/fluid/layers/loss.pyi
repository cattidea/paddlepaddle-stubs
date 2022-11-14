from __future__ import annotations

from typing import Any, Optional

def center_loss(input: Any, label: Any, num_classes: Any, alpha: Any, param_attr: Any, update_center: bool = ...): ...
def bpr_loss(input: Any, label: Any, name: str | None = ...): ...
def cross_entropy(input: Any, label: Any, soft_label: bool = ..., ignore_index: Any = ...): ...
def square_error_cost(input: Any, label: Any): ...
def edit_distance(
    input: Any,
    label: Any,
    normalized: bool = ...,
    ignored_tokens: Any | None = ...,
    input_length: Any | None = ...,
    label_length: Any | None = ...,
): ...
def warpctc(
    input: Any,
    label: Any,
    blank: int = ...,
    norm_by_times: bool = ...,
    input_length: Any | None = ...,
    label_length: Any | None = ...,
): ...
def nce(
    input: Any,
    label: Any,
    num_total_classes: Any,
    sample_weight: Any | None = ...,
    param_attr: Any | None = ...,
    bias_attr: Any | None = ...,
    num_neg_samples: Any | None = ...,
    name: str | None = ...,
    sampler: str = ...,
    custom_dist: Any | None = ...,
    seed: int = ...,
    is_sparse: bool = ...,
): ...
def hsigmoid(
    input: Any,
    label: Any,
    num_classes: Any,
    param_attr: Any | None = ...,
    bias_attr: Any | None = ...,
    name: str | None = ...,
    path_table: Any | None = ...,
    path_code: Any | None = ...,
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
    customized_samples: Any | None = ...,
    customized_probabilities: Any | None = ...,
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
def rank_loss(label: Any, left: Any, right: Any, name: str | None = ...): ...
def margin_rank_loss(label: Any, left: Any, right: Any, margin: float = ..., name: str | None = ...): ...
def sigmoid_cross_entropy_with_logits(
    x: Any, label: Any, ignore_index: Any = ..., name: str | None = ..., normalize: bool = ...
): ...
def teacher_student_sigmoid_loss(
    input: Any, label: Any, soft_max_up_bound: float = ..., soft_max_lower_bound: Any = ...
): ...
def huber_loss(input: Any, label: Any, delta: Any): ...
def kldiv_loss(x: Any, target: Any, reduction: str = ..., name: str | None = ...): ...
def npair_loss(anchor: Any, positive: Any, labels: Any, l2_reg: float = ...): ...
def mse_loss(input: Any, label: Any): ...
