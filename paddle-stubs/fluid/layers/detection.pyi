from __future__ import annotations

from typing import Any, Optional

def retinanet_target_assign(
    bbox_pred: Any,
    cls_logits: Any,
    anchor_box: Any,
    anchor_var: Any,
    gt_boxes: Any,
    gt_labels: Any,
    is_crowd: Any,
    im_info: Any,
    num_classes: int = ...,
    positive_overlap: float = ...,
    negative_overlap: float = ...,
): ...
def rpn_target_assign(
    bbox_pred: Any,
    cls_logits: Any,
    anchor_box: Any,
    anchor_var: Any,
    gt_boxes: Any,
    is_crowd: Any,
    im_info: Any,
    rpn_batch_size_per_im: int = ...,
    rpn_straddle_thresh: float = ...,
    rpn_fg_fraction: float = ...,
    rpn_positive_overlap: float = ...,
    rpn_negative_overlap: float = ...,
    use_random: bool = ...,
): ...
def sigmoid_focal_loss(x: Any, label: Any, fg_num: Any, gamma: float = ..., alpha: float = ...): ...
def detection_output(
    loc: Any,
    scores: Any,
    prior_box: Any,
    prior_box_var: Any,
    background_label: int = ...,
    nms_threshold: float = ...,
    nms_top_k: int = ...,
    keep_top_k: int = ...,
    score_threshold: float = ...,
    nms_eta: float = ...,
    return_index: bool = ...,
): ...
def iou_similarity(x: Any, y: Any, box_normalized: bool = ..., name: Optional[str] = ...): ...
def box_coder(
    prior_box: Any,
    prior_box_var: Any,
    target_box: Any,
    code_type: str = ...,
    box_normalized: bool = ...,
    name: Optional[str] = ...,
    axis: int = ...,
): ...
def polygon_box_transform(input: Any, name: Optional[str] = ...): ...
def yolov3_loss(
    x: Any,
    gt_box: Any,
    gt_label: Any,
    anchors: Any,
    anchor_mask: Any,
    class_num: Any,
    ignore_thresh: Any,
    downsample_ratio: Any,
    gt_score: Optional[Any] = ...,
    use_label_smooth: bool = ...,
    name: Optional[str] = ...,
    scale_x_y: float = ...,
): ...
def yolo_box(
    x: Any,
    img_size: Any,
    anchors: Any,
    class_num: Any,
    conf_thresh: Any,
    downsample_ratio: Any,
    clip_bbox: bool = ...,
    name: Optional[str] = ...,
    scale_x_y: float = ...,
    iou_aware: bool = ...,
    iou_aware_factor: float = ...,
): ...
def bipartite_match(
    dist_matrix: Any, match_type: Optional[Any] = ..., dist_threshold: Optional[Any] = ..., name: Optional[str] = ...
): ...
def target_assign(
    input: Any,
    matched_indices: Any,
    negative_indices: Optional[Any] = ...,
    mismatch_value: Optional[Any] = ...,
    name: Optional[str] = ...,
): ...
def ssd_loss(
    location: Any,
    confidence: Any,
    gt_box: Any,
    gt_label: Any,
    prior_box: Any,
    prior_box_var: Optional[Any] = ...,
    background_label: int = ...,
    overlap_threshold: float = ...,
    neg_pos_ratio: float = ...,
    neg_overlap: float = ...,
    loc_loss_weight: float = ...,
    conf_loss_weight: float = ...,
    match_type: str = ...,
    mining_type: str = ...,
    normalize: bool = ...,
    sample_size: Optional[Any] = ...,
): ...
def prior_box(
    input: Any,
    image: Any,
    min_sizes: Any,
    max_sizes: Optional[Any] = ...,
    aspect_ratios: Any = ...,
    variance: Any = ...,
    flip: bool = ...,
    clip: bool = ...,
    steps: Any = ...,
    offset: float = ...,
    name: Optional[str] = ...,
    min_max_aspect_ratios_order: bool = ...,
): ...
def density_prior_box(
    input: Any,
    image: Any,
    densities: Optional[Any] = ...,
    fixed_sizes: Optional[Any] = ...,
    fixed_ratios: Optional[Any] = ...,
    variance: Any = ...,
    clip: bool = ...,
    steps: Any = ...,
    offset: float = ...,
    flatten_to_2d: bool = ...,
    name: Optional[str] = ...,
): ...
def multi_box_head(
    inputs: Any,
    image: Any,
    base_size: Any,
    num_classes: Any,
    aspect_ratios: Any,
    min_ratio: Optional[Any] = ...,
    max_ratio: Optional[Any] = ...,
    min_sizes: Optional[Any] = ...,
    max_sizes: Optional[Any] = ...,
    steps: Optional[Any] = ...,
    step_w: Optional[Any] = ...,
    step_h: Optional[Any] = ...,
    offset: float = ...,
    variance: Any = ...,
    flip: bool = ...,
    clip: bool = ...,
    kernel_size: int = ...,
    pad: int = ...,
    stride: int = ...,
    name: Optional[str] = ...,
    min_max_aspect_ratios_order: bool = ...,
): ...
def anchor_generator(
    input: Any,
    anchor_sizes: Optional[Any] = ...,
    aspect_ratios: Optional[Any] = ...,
    variance: Any = ...,
    stride: Optional[Any] = ...,
    offset: float = ...,
    name: Optional[str] = ...,
): ...
def roi_perspective_transform(
    input: Any,
    rois: Any,
    transformed_height: Any,
    transformed_width: Any,
    spatial_scale: float = ...,
    name: Optional[str] = ...,
): ...
def generate_proposal_labels(
    rpn_rois: Any,
    gt_classes: Any,
    is_crowd: Any,
    gt_boxes: Any,
    im_info: Any,
    batch_size_per_im: int = ...,
    fg_fraction: float = ...,
    fg_thresh: float = ...,
    bg_thresh_hi: float = ...,
    bg_thresh_lo: float = ...,
    bbox_reg_weights: Any = ...,
    class_nums: Optional[Any] = ...,
    use_random: bool = ...,
    is_cls_agnostic: bool = ...,
    is_cascade_rcnn: bool = ...,
    max_overlap: Optional[Any] = ...,
    return_max_overlap: bool = ...,
): ...
def generate_mask_labels(
    im_info: Any,
    gt_classes: Any,
    is_crowd: Any,
    gt_segms: Any,
    rois: Any,
    labels_int32: Any,
    num_classes: Any,
    resolution: Any,
): ...
def generate_proposals(
    scores: Any,
    bbox_deltas: Any,
    im_info: Any,
    anchors: Any,
    variances: Any,
    pre_nms_top_n: int = ...,
    post_nms_top_n: int = ...,
    nms_thresh: float = ...,
    min_size: float = ...,
    eta: float = ...,
    return_rois_num: bool = ...,
    name: Optional[str] = ...,
): ...
def box_clip(input: Any, im_info: Any, name: Optional[str] = ...): ...
def retinanet_detection_output(
    bboxes: Any,
    scores: Any,
    anchors: Any,
    im_info: Any,
    score_threshold: float = ...,
    nms_top_k: int = ...,
    keep_top_k: int = ...,
    nms_threshold: float = ...,
    nms_eta: float = ...,
): ...
def multiclass_nms(
    bboxes: Any,
    scores: Any,
    score_threshold: Any,
    nms_top_k: Any,
    keep_top_k: Any,
    nms_threshold: float = ...,
    normalized: bool = ...,
    nms_eta: float = ...,
    background_label: int = ...,
    name: Optional[str] = ...,
): ...
def locality_aware_nms(
    bboxes: Any,
    scores: Any,
    score_threshold: Any,
    nms_top_k: Any,
    keep_top_k: Any,
    nms_threshold: float = ...,
    normalized: bool = ...,
    nms_eta: float = ...,
    background_label: int = ...,
    name: Optional[str] = ...,
): ...
def matrix_nms(
    bboxes: Any,
    scores: Any,
    score_threshold: Any,
    post_threshold: Any,
    nms_top_k: Any,
    keep_top_k: Any,
    use_gaussian: bool = ...,
    gaussian_sigma: float = ...,
    background_label: int = ...,
    normalized: bool = ...,
    return_index: bool = ...,
    name: Optional[str] = ...,
): ...
def distribute_fpn_proposals(
    fpn_rois: Any,
    min_level: Any,
    max_level: Any,
    refer_level: Any,
    refer_scale: Any,
    rois_num: Optional[Any] = ...,
    name: Optional[str] = ...,
): ...
def box_decoder_and_assign(
    prior_box: Any, prior_box_var: Any, target_box: Any, box_score: Any, box_clip: Any, name: Optional[str] = ...
): ...
def collect_fpn_proposals(
    multi_rois: Any,
    multi_scores: Any,
    min_level: Any,
    max_level: Any,
    post_nms_top_n: Any,
    rois_num_per_level: Optional[Any] = ...,
    name: Optional[str] = ...,
): ...
