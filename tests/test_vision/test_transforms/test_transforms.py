# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

from typing import Any

import numpy as np
import paddle
from paddle import Tensor
from PIL import Image
from PIL.Image import Image as PILImage
from typing_extensions import assert_type


def test_import():
    paddle.vision.transforms.Compose
    paddle.vision.transforms.BaseTransform
    paddle.vision.transforms.ToTensor
    paddle.vision.transforms.Resize
    paddle.vision.transforms.RandomResizedCrop
    paddle.vision.transforms.CenterCrop
    paddle.vision.transforms.RandomHorizontalFlip
    paddle.vision.transforms.RandomVerticalFlip
    paddle.vision.transforms.Normalize
    paddle.vision.transforms.Transpose
    paddle.vision.transforms.BrightnessTransform
    paddle.vision.transforms.ContrastTransform
    paddle.vision.transforms.SaturationTransform
    paddle.vision.transforms.HueTransform
    paddle.vision.transforms.ColorJitter
    paddle.vision.transforms.RandomCrop
    paddle.vision.transforms.Pad
    paddle.vision.transforms.RandomRotation
    paddle.vision.transforms.Grayscale

    from paddle.vision.transforms import (
        BaseTransform,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.vision.transforms import (
        BrightnessTransform,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.vision.transforms import (
        CenterCrop,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.vision.transforms import (
        ColorJitter,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.vision.transforms import Compose  # pyright: ignore [reportUnusedImport]
    from paddle.vision.transforms import (
        ContrastTransform,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.vision.transforms import (
        Grayscale,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.vision.transforms import (
        HueTransform,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.vision.transforms import (
        Normalize,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.vision.transforms import Pad  # pyright: ignore [reportUnusedImport]
    from paddle.vision.transforms import (
        RandomCrop,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.vision.transforms import (
        RandomHorizontalFlip,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.vision.transforms import (
        RandomResizedCrop,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.vision.transforms import (
        RandomRotation,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.vision.transforms import (
        RandomVerticalFlip,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.vision.transforms import Resize  # pyright: ignore [reportUnusedImport]
    from paddle.vision.transforms import (
        SaturationTransform,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.vision.transforms import (
        ToTensor,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.vision.transforms import (
        Transpose,  # pyright: ignore [reportUnusedImport]
    )


def test_types():
    img_tensor = paddle.randint(0, 255, shape=[3, 224, 224]).astype(paddle.uint8)
    img_np = img_tensor.numpy()
    img_pil = Image.fromarray(img_np).convert("RGB")  # type: ignore

    transform = paddle.vision.transforms.ToTensor()
    assert_type(transform, paddle.vision.transforms.ToTensor)
    assert_type(transform(img_pil), Tensor)
    assert_type(transform(img_np), Tensor)

    transform = paddle.vision.transforms.Resize(256)
    assert_type(transform, paddle.vision.transforms.Resize)
    assert_type(transform(img_pil), PILImage)
    assert_type(transform(img_np), np.ndarray[Any, Any])
    assert_type(transform(img_tensor), Tensor)

    transform = paddle.vision.transforms.RandomResizedCrop(224)
    assert_type(transform, paddle.vision.transforms.RandomResizedCrop)
    assert_type(transform(img_pil), PILImage)
    assert_type(transform(img_np), np.ndarray[Any, Any])
    assert_type(transform(img_tensor), Tensor)

    transform = paddle.vision.transforms.CenterCrop(224)
    assert_type(transform, paddle.vision.transforms.CenterCrop)
    assert_type(transform(img_pil), PILImage)
    assert_type(transform(img_np), np.ndarray[Any, Any])

    transform = paddle.vision.transforms.RandomHorizontalFlip()
    assert_type(transform, paddle.vision.transforms.RandomHorizontalFlip)
    assert_type(transform(img_pil), PILImage)
    assert_type(transform(img_np), np.ndarray[Any, Any])
    assert_type(transform(img_tensor), Tensor)

    transform = paddle.vision.transforms.RandomVerticalFlip()
    assert_type(transform, paddle.vision.transforms.RandomVerticalFlip)
    assert_type(transform(img_pil), PILImage)
    assert_type(transform(img_np), np.ndarray[Any, Any])

    transform = paddle.vision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    assert_type(transform, paddle.vision.transforms.Normalize)
    assert_type(transform(img_pil), PILImage)
    assert_type(transform(img_np), np.ndarray[Any, Any])

    transform = paddle.vision.transforms.Transpose([0, 2, 1])
    assert_type(transform, paddle.vision.transforms.Transpose)
    assert_type(transform(img_pil), PILImage)
    assert_type(transform(img_np), np.ndarray[Any, Any])

    transform = paddle.vision.transforms.BrightnessTransform(0.1)
    assert_type(transform, paddle.vision.transforms.BrightnessTransform)
    assert_type(transform(img_pil), PILImage)
    assert_type(transform(img_np), np.ndarray[Any, Any])

    transform = paddle.vision.transforms.ContrastTransform(0.1)
    assert_type(transform, paddle.vision.transforms.ContrastTransform)
    assert_type(transform(img_pil), PILImage)
    assert_type(transform(img_np), np.ndarray[Any, Any])

    transform = paddle.vision.transforms.SaturationTransform(0.1)
    assert_type(transform, paddle.vision.transforms.SaturationTransform)
    assert_type(transform(img_pil), PILImage)
    assert_type(transform(img_np), np.ndarray[Any, Any])

    transform = paddle.vision.transforms.HueTransform(0.1)
    assert_type(transform, paddle.vision.transforms.HueTransform)
    assert_type(transform(img_pil), PILImage)
    assert_type(transform(img_np), np.ndarray[Any, Any])

    transform = paddle.vision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    assert_type(transform, paddle.vision.transforms.ColorJitter)
    assert_type(transform(img_pil), PILImage)
    assert_type(transform(img_np), np.ndarray[Any, Any])

    transform = paddle.vision.transforms.Pad(padding=10, fill=0)
    assert_type(transform, paddle.vision.transforms.Pad)
    assert_type(transform(img_pil), PILImage)
    assert_type(transform(img_np), np.ndarray[Any, Any])

    transform = paddle.vision.transforms.RandomRotation(10)
    assert_type(transform, paddle.vision.transforms.RandomRotation)
    assert_type(transform(img_pil), PILImage)
    assert_type(transform(img_np), np.ndarray[Any, Any])

    transform = paddle.vision.transforms.Grayscale()
    assert_type(transform, paddle.vision.transforms.Grayscale)
    assert_type(transform(img_pil), PILImage)
    assert_type(transform(img_np), np.ndarray[Any, Any])
    assert_type(transform(img_tensor), Tensor)

    transform = paddle.vision.transforms.Compose(
        [
            paddle.vision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            paddle.vision.transforms.Transpose([0, 2, 1]),
        ]
    )
    assert_type(transform, paddle.vision.transforms.Compose)
    assert_type(transform(img_pil), PILImage)
    assert_type(transform(img_np), np.ndarray[Any, Any])
    assert_type(transform(img_tensor), Tensor)
