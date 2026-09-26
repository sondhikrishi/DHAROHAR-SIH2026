"""
Visualization utilities for IndicOCR
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional, Dict, Any, Union
import colorsys
import logging

logger = logging.getLogger(__name__)

# Type aliases
CvImage = np.ndarray
Color = Tuple[int, int, int]  # BGR format


# Predefined color palette for different languages/models
COLOR_PALETTE = [
    (0, 255, 0),      # Green
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 128),    # Purple
    (255, 165, 0),    # Orange
    (0, 128, 128),    # Teal
    (128, 128, 0),    # Olive
    (255, 192, 203),  # Pink
    (0, 0, 128),      # Navy
]

LANGUAGE_COLORS = {
    "hi": (0, 255, 0),
    "bn": (255, 0, 0),
    "ta": (0, 0, 255),
    "te": (255, 255, 0),
    "kn": (255, 0, 255),
    "ml": (0, 255, 255),
    "mr": (128, 0, 128),
    "gu": (255, 165, 0),
    "pa": (0, 128, 128),
    "or": (128, 128, 0),
    "as": (255, 192, 203),
    "ur": (0, 0, 128),
    "en": (128, 128, 128),
    "auto": (255, 255, 255),
}

MODEL_COLORS = {
    "paddle_ocr": (0, 255, 0),
    "trocr": (255, 0, 0),
    "tesseract": (0, 0, 255),
    "easyocr": (255, 255, 0),
    "ensemble": (255, 0, 255),
}


def get_color_for_language(lang: str) -> Color:
    """Get color for a language code"""
    return LANGUAGE_COLORS.get(lang.lower(), (255, 255, 255))


def get_color_for_model(model: str) -> Color:
    """Get color for a model name"""
    return MODEL_COLORS.get(model.lower(), (255, 255, 255))


def generate_distinct_colors(n: int) -> List[Color]:
    """Generate n visually distinct colors"""
    colors = []
    for i in range(n):
        hue = i / n
        sat = 0.8 + (i % 2) * 0.2
        val = 0.9
        rgb = colorsys.hsv_to_rgb(hue, sat, val)
        colors.append(tuple(int(c * 255) for c in rgb[::-1]))  # Convert to BGR
    return colors


def draw_boxes(
    image: CvImage,
    boxes: List[Union[List[float], np.ndarray]],
    color: Color = (0, 255, 0),
    thickness: int = 2,
    labels: Optional[List[str]] = None,
    font_scale: float = 0.6,
    font_thickness: int = 1
) -> CvImage:
    """
    Draw bounding boxes on image
    
    Args:
        image: Input image
        boxes: List of boxes [x1, y1, x2, y2] or polygons [[x1,y1], [x2,y2], ...]
        color: Box color (BGR)
        thickness: Line thickness
        labels: Optional labels for each box
        font_scale: Font scale for labels
        font_thickness: Font thickness
        
    Returns:
        Image with boxes drawn
    """
    img = image.copy()
    
    for i, box in enumerate(boxes):
        box = np.array(box, dtype=np.int32)
        
        if len(box.shape) == 1 and len(box) == 4:
            # Rectangle format [x1, y1, x2, y2]
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
            
            if labels and i < len(labels):
                label = labels[i]
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        elif len(box.shape) == 2:
            # Polygon format
            cv2.polylines(img, [box], True, color, thickness)
            
            if labels and i < len(labels):
                label = labels[i]
                x1, y1 = box[0]
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
                cv2.putText(img, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
    
    return img


def draw_text_regions(
    image: CvImage,
    regions: List[Dict[str, Any]],
    color_by: str = "language",
    show_confidence: bool = True,
    show_language: bool = True,
    font_scale: float = 0.5
) -> CvImage:
    """
    Draw text regions with text overlay
    
    Args:
        image: Input image
        regions: List of region dicts with keys: bbox/polygon, text, confidence, language, model_source
        color_by: "language", "model", or "confidence"
        show_confidence: Whether to show confidence score
        show_language: Whether to show language code
        font_scale: Font scale
        
    Returns:
        Annotated image
    """
    img = image.copy()
    
    for region in regions:
        # Get bbox
        if "polygon" in region and region["polygon"]:
            box = np.array(region["polygon"], dtype=np.int32)
            x1, y1 = box[:, 0].min(), box[:, 1].min()
        elif "bbox" in region:
            box = np.array(region["bbox"], dtype=np.int32)
            if len(box) == 4:
                x1, y1, x2, y2 = box
                box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
            else:
                continue
            x1, y1 = box[0]
        else:
            continue
        
        # Determine color
        if color_by == "language":
            color = get_color_for_language(region.get("language", "auto"))
        elif color_by == "model":
            color = get_color_for_model(region.get("model_source", "unknown"))
        elif color_by == "confidence":
            conf = region.get("confidence", 0)
            # Green to red gradient
            color = (0, int(255 * conf), int(255 * (1 - conf)))
        else:
            color = (0, 255, 0)
        
        # Draw polygon
        cv2.polylines(img, [box], True, color, 2)
        
        # Prepare label
        label_parts = []
        if show_confidence:
            label_parts.append(f"{region.get('confidence', 0):.2f}")
        if show_language:
            label_parts.append(region.get("language", "??"))
        
        if label_parts:
            label = " | ".join(label_parts)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
        
        # Draw text (truncated)
        text = region.get("text", "")
        if text:
            display_text = text[:50] + "..." if len(text) > 50 else text
            (tw, th), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, 1)
            text_y = y1 + th + 4
            cv2.rectangle(img, (x1, text_y - th - 2), (x1 + tw + 4, text_y + 2), (0, 0, 0), -1)
            cv2.putText(img, display_text, (x1 + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, color, 1)
    
    return img

def draw_text(
    image: CvImage,
    text: str,
    position: Tuple[int, int],
    color: Color = (0, 255, 0),
    font_scale: float = 0.5,
    thickness: int = 1
) -> CvImage:
    """Draw text on an image."""
    img = image.copy()

    cv2.putText(
        img,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )

    return img


def create_comparison_visualization(
    image: CvImage,
    model_results: Dict[str, List[Dict[str, Any]]],
    ensemble_result: Optional[List[Dict[str, Any]]] = None
) -> CvImage:
    """
    Create side-by-side comparison of different model results
    
    Args:
        image: Original image
        model_results: Dict of model_name -> list of regions
        ensemble_result: Optional ensemble result
        
    Returns:
        Comparison visualization
    """
    h, w = image.shape[:2]
    n_models = len(model_results) + (1 if ensemble_result else 0)
    
    # Create grid
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    cell_w = w
    cell_h = h
    
    canvas = np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) * 255
    
    all_results = list(model_results.items())
    if ensemble_result:
        all_results.append(("Ensemble", ensemble_result))
    
    for idx, (model_name, regions) in enumerate(all_results):
        row = idx // cols
        col = idx % cols
        
        y_start = row * cell_h
        x_start = col * cell_w
        
        # Place original image
        cell = image.copy()
        
        # Draw regions
        color = get_color_for_model(model_name)
        cell = draw_text_regions(cell, regions, color_by="model", show_confidence=True, show_language=True)
        
        # Add model name header
        cv2.rectangle(cell, (0, 0), (w, 40), color, -1)
        cv2.putText(cell, model_name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Place on canvas
        canvas[y_start:y_start+cell_h, x_start:x_start+cell_w] = cell
    
    return canvas


def create_visualization(
    image: CvImage,
    regions: List[Dict[str, Any]],
    color_by: str = "language",
    show_confidence: bool = True,
    show_language: bool = True,
    show_text: bool = True
) -> CvImage:
    """
    Main visualization function
    
    Returns:
        Annotated image
    """
    return draw_text_regions(
        image,
        regions,
        color_by=color_by,
        show_confidence=show_confidence,
        show_language=show_language
    )


def save_visualization(
    image: CvImage,
    path: str,
    quality: int = 95
) -> bool:
    """Save visualization to file"""
    try:
        if path.lower().endswith(('.jpg', '.jpeg')):
            return cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            return cv2.imwrite(path, image)
    except Exception as e:
        logger.error(f"Failed to save visualization: {e}")
        return False


def draw_legend(
    image: CvImage,
    items: List[Tuple[str, Color]],
    position: str = "top-right",
    padding: int = 10,
    font_scale: float = 0.5
) -> CvImage:
    """Draw color legend on image"""
    img = image.copy()
    h, w = img.shape[:2]
    
    item_height = 25
    max_width = max(cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0] for name, _ in items) + 50
    legend_w = max_width + padding * 2
    legend_h = len(items) * item_height + padding * 2
    
    if position == "top-right":
        x1, y1 = w - legend_w - padding, padding
    elif position == "top-left":
        x1, y1 = padding, padding
    elif position == "bottom-right":
        x1, y1 = w - legend_w - padding, h - legend_h - padding
    elif position == "bottom-left":
        x1, y1 = padding, h - legend_h - padding
    else:
        x1, y1 = padding, padding
    
    x2, y2 = x1 + legend_w, y1 + legend_h
    
    # Background
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
    
    # Items
    for i, (name, color) in enumerate(items):
        y = y1 + padding + i * item_height
        cv2.rectangle(img, (x1 + padding, y), (x1 + padding + 20, y + 18), color, -1)
        cv2.putText(img, name, (x1 + padding + 25, y + 14), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    
    return img


def create_confidence_heatmap(
    image: CvImage,
    regions: List[Dict[str, Any]],
    alpha: float = 0.5
) -> CvImage:
    """Create confidence heatmap overlay"""
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    for region in regions:
        conf = region.get("confidence", 0)
        if "polygon" in region and region["polygon"]:
            pts = np.array(region["polygon"], dtype=np.int32)
            cv2.fillPoly(heatmap, [pts], conf)
        elif "bbox" in region:
            x1, y1, x2, y2 = map(int, region["bbox"])
            cv2.rectangle(heatmap, (x1, y1), (x2, y2), conf, -1)
    
    # Normalize and apply colormap
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blend
    result = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)
    return result