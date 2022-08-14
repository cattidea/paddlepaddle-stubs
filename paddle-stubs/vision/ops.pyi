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
def deform_conv2d(
    x: Any,
    offset: Any,
    weight: Any,
    bias: Optional[Any] = ...,
    stride: int = ...,
    padding: int = ...,
    dilation: int = ...,
    deformable_groups: int = ...,
    groups: int = ...,
    mask: Optional[Any] = ...,
    name: Optional[str] = ...,
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
        weight_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
    ): ...
    def forward(self, x: Any, offset: Any, mask: Optional[Any] = ...): ...

def read_file(filename: Any, name: Optional[str] = ...): ...
def decode_jpeg(x: Any, mode: str = ..., name: Optional[str] = ...): ...
def psroi_pool(
    x: Any, boxes: Any, boxes_num: Any, output_size: Any, spatial_scale: float = ..., name: Optional[str] = ...
): ...

class PSRoIPool(Layer):
    output_size: Any = ...
    spatial_scale: Any = ...
    def __init__(self, output_size: Any, spatial_scale: float = ...) -> None: ...
    def forward(self, x: Any, boxes: Any, boxes_num: Any): ...

def roi_pool(
    x: Any, boxes: Any, boxes_num: Any, output_size: Any, spatial_scale: float = ..., name: Optional[str] = ...
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
    name: Optional[str] = ...,
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
        padding: Optional[Any] = ...,
        groups: int = ...,
        norm_layer: Any = ...,
        activation_layer: Any = ...,
        dilation: int = ...,
        bias: Optional[Any] = ...,
    ) -> None: ...

def nms(
    boxes: Any,
    iou_threshold: float = ...,
    scores: Optional[Any] = ...,
    category_idxs: Optional[Any] = ...,
    categories: Optional[Any] = ...,
    top_k: Optional[Any] = ...,
): ...
