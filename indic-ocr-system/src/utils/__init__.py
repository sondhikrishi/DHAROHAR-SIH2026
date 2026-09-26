"""
Utility modules for IndicOCR
"""

from .image_utils import (
    load_image,
    save_image,
    resize_image,
    crop_image,
    rotate_image,
    deskew_image,
    denoise_image,
    enhance_contrast,
    binarize_image,
    auto_orient,
    image_to_base64,
    base64_to_image
)

from .visualization import (
    draw_boxes,
    draw_text,
    create_visualization,
    save_visualization
)

from .metrics import (
    calculate_cer,
    calculate_wer,
    calculate_f1,
    compute_metrics
)

__all__ = [
    # image_utils
    "load_image",
    "save_image",
    "resize_image",
    "crop_image",
    "rotate_image",
    "deskew_image",
    "denoise_image",
    "enhance_contrast",
    "binarize_image",
    "auto_orient",
    "image_to_base64",
    "base64_to_image",
    # visualization
    "draw_boxes",
    "draw_text",
    "create_visualization",
    "save_visualization",
    # metrics
    "calculate_cer",
    "calculate_wer",
    "calculate_f1",
    "compute_metrics",
]