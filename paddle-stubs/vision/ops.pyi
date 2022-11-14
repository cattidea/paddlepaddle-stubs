from __future__ import annotations

from typing import Any, Optional

from paddle.common_ops_import import *

from ..nn import Layer, Sequential

def yolo_loss(
    x: Any,
    gt_box: Any,
    gt_label: Any,
    anchors: Any,
    anchor_mask: Any,
    class_num: Any,
    ignore_thresh: Any,
    downsample_ratio: Any,
    gt_score: Any | None = ...,
    use_label_smooth: bool = ...,
    name: str | None = ...,
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
    name: str | None = ...,
    scale_x_y: float = ...,
    iou_aware: bool = ...,
    iou_aware_factor: float = ...,
): ...
def deform_conv2d(
    x: Any,
    offset: Any,
    weight: Any,
    bias: Any | None = ...,
    stride: int = ...,
    padding: int = ...,
    dilation: int = ...,
    deformable_groups: int = ...,
    groups: int = ...,
    mask: Any | None = ...,
    name: str | None = ...,
): ...

class DeformConv2D(Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        in_channels: Any,
        out_channels: Any,
        kernel_size: Any,
        stride: int = ...,
        padding: int = ...,
        dilation: int = ...,
        deformable_groups: int = ...,
        groups: int = ...,
        weight_attr: Any | None = ...,
        bias_attr: Any | None = ...,
    ): ...
    def forward(self, x: Any, offset: Any, mask: Any | None = ...): ...

def read_file(filename: Any, name: str | None = ...): ...
def decode_jpeg(x: Any, mode: str = ..., name: str | None = ...): ...
def psroi_pool(
    x: Any, boxes: Any, boxes_num: Any, output_size: Any, spatial_scale: float = ..., name: str | None = ...
): ...

class PSRoIPool(Layer):
    output_size: Any = ...
    spatial_scale: Any = ...
    def __init__(self, output_size: Any, spatial_scale: float = ...) -> None: ...
    def forward(self, x: Any, boxes: Any, boxes_num: Any): ...

def roi_pool(
    x: Any, boxes: Any, boxes_num: Any, output_size: Any, spatial_scale: float = ..., name: str | None = ...
): ...

class RoIPool(Layer):
    def __init__(self, output_size: Any, spatial_scale: float = ...) -> None: ...
    def forward(self, x: Any, boxes: Any, boxes_num: Any): ...
    def extra_repr(self): ...

def roi_align(
    x: Any,
    boxes: Any,
    boxes_num: Any,
    output_size: Any,
    spatial_scale: float = ...,
    sampling_ratio: int = ...,
    aligned: bool = ...,
    name: str | None = ...,
): ...

class RoIAlign(Layer):
    def __init__(self, output_size: Any, spatial_scale: float = ...) -> None: ...
    def forward(self, x: Any, boxes: Any, boxes_num: Any, aligned: bool = ...): ...

class ConvNormActivation(Sequential):
    def __init__(
        self,
        in_channels: Any,
        out_channels: Any,
        kernel_size: int = ...,
        stride: int = ...,
        padding: Any | None = ...,
        groups: int = ...,
        norm_layer: Any = ...,
        activation_layer: Any = ...,
        dilation: int = ...,
        bias: Any | None = ...,
    ) -> None: ...

def nms(
    boxes: Any,
    iou_threshold: float = ...,
    scores: Any | None = ...,
    category_idxs: Any | None = ...,
    categories: Any | None = ...,
    top_k: Any | None = ...,
): ...
