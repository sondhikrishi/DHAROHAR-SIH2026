"""
Image processing utilities for IndicOCR
"""

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from typing import Tuple, Optional, Union, List
import base64
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

# Type aliases
ImageType = Union[np.ndarray, Image.Image, str, bytes]
CvImage = np.ndarray
PilImage = Image.Image


def load_image(image: ImageType) -> CvImage:
    """
    Load image from various sources to OpenCV format (BGR)
    
    Args:
        image: Path (str), bytes, PIL Image, or numpy array
        
    Returns:
        OpenCV image (BGR numpy array)
    """
    if isinstance(image, str):
        # File path
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Could not load image from path: {image}")
        return img
    
    elif isinstance(image, bytes):
        # Bytes
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from bytes")
        return img
    
    elif isinstance(image, Image.Image):
        # PIL Image -> OpenCV
        return pil_to_cv(image)
    
    elif isinstance(image, np.ndarray):
        # Already numpy array
        if len(image.shape) == 2:
            # Grayscale -> BGR
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            # RGBA -> BGR
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        elif image.shape[2] == 3:
            # Assume BGR (OpenCV default)
            return image
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
    
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")


def save_image(image: CvImage, path: str, quality: int = 95) -> bool:
    """Save OpenCV image to file"""
    try:
        if path.lower().endswith(('.jpg', '.jpeg')):
            return cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        elif path.lower().endswith('.png'):
            return cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            return cv2.imwrite(path, image)
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        return False


def pil_to_cv(pil_img: PilImage) -> CvImage:
    """Convert PIL Image (RGB) to OpenCV (BGR)"""
    if pil_img.mode == 'RGBA':
        # White background for transparency
        background = Image.new('RGB', pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])
        pil_img = background
    elif pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv_to_pil(cv_img: CvImage) -> PilImage:
    """Convert OpenCV (BGR) to PIL Image (RGB)"""
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def resize_image(
    image: CvImage,
    max_dimension: int = 3200,
    min_dimension: int = 640,
    keep_aspect_ratio: bool = True
) -> CvImage:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        max_dimension: Maximum width or height
        min_dimension: Minimum width or height
        keep_aspect_ratio: Whether to preserve aspect ratio
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    # Check if resize needed
    if max_dimension and (w > max_dimension or h > max_dimension):
        if keep_aspect_ratio:
            scale = min(max_dimension / w, max_dimension / h)
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            new_w, new_h = max_dimension, max_dimension
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    if min_dimension and (w < min_dimension or h < min_dimension):
        if keep_aspect_ratio:
            scale = max(min_dimension / w, min_dimension / h)
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            new_w, new_h = min_dimension, min_dimension
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    return image


def crop_image(image: CvImage, bbox: Tuple[int, int, int, int]) -> CvImage:
    """Crop image using bounding box (x1, y1, x2, y2)"""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    
    # Clamp to image bounds
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    if x2 <= x1 or y2 <= y1:
        return image
    
    return image[y1:y2, x1:x2]


def rotate_image(image: CvImage, angle: float, center: Optional[Tuple[int, int]] = None) -> CvImage:
    """Rotate image by angle (degrees)"""
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def deskew_image(image: CvImage, max_angle: float = 45.0) -> Tuple[CvImage, float]:
    """
    Deskew image using minAreaRect on text contours
    
    Returns:
        (deskewed_image, angle_corrected)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image, 0.0
    
    # Filter contours by area
    contours = [c for c in contours if cv2.contourArea(c) > 100]
    
    if not contours:
        return image, 0.0
    
    # Get minimum area rectangle for largest contour
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    angle = rect[-1]
    
    # Normalize angle
    if angle < -45:
        angle = 90 + angle
    
    if abs(angle) > max_angle:
        angle = 0.0
    
    if abs(angle) > 0.5:
        rotated = rotate_image(image, angle)
        return rotated, angle
    
    return image, 0.0


def denoise_image(
    image: CvImage,
    method: str = "nlm",
    strength: float = 10.0
) -> CvImage:
    """
    Denoise image
    
    Methods:
        - nlm: Non-local Means (best quality, slower)
        - gaussian: Gaussian blur (fast)
        - bilateral: Bilateral filter (preserves edges)
        - median: Median filter (good for salt-and-pepper)
    """
    if method == "nlm":
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    elif method == "gaussian":
        ksize = int(strength) | 1  # Ensure odd
        return cv2.GaussianBlur(image, (ksize, ksize), 0)
    elif method == "bilateral":
        return cv2.bilateralFilter(image, 9, strength * 7, strength * 7)
    elif method == "median":
        ksize = int(strength) | 1
        return cv2.medianBlur(image, ksize)
    else:
        logger.warning(f"Unknown denoise method: {method}, using NLM")
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def enhance_contrast(
    image: CvImage,
    method: str = "clahe",
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8)
) -> CvImage:
    """
    Enhance image contrast
    
    Methods:
        - clahe: Contrast Limited Adaptive Histogram Equalization
        - histogram: Global histogram equalization
        - pil: PIL-based enhancement
    """
    if method == "clahe":
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    elif method == "histogram":
        if len(image.shape) == 3:
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            return cv2.equalizeHist(image)
    
    elif method == "pil":
        pil_img = cv_to_pil(image)
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.5)
        return pil_to_cv(pil_img)
    
    return image


def binarize_image(
    image: CvImage,
    method: str = "sauvola",
    block_size: int = 15,
    k: float = 0.2
) -> CvImage:
    """
    Binarize image
    
    Methods:
        - sauvola: Sauvola adaptive threshold (best for documents)
        - otsu: Otsu's global threshold
        - adaptive: OpenCV adaptive threshold
        - niblack: Niblack threshold
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    if method == "sauvola":
        # Sauvola implementation
        from skimage.filters import threshold_sauvola
        thresh = threshold_sauvola(gray, window_size=block_size, k=k)
        binary = (gray > thresh).astype(np.uint8) * 255
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    elif method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, 10
        )
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    elif method == "niblack":
        from skimage.filters import threshold_niblack
        thresh = threshold_niblack(gray, window_size=block_size, k=k)
        binary = (gray > thresh).astype(np.uint8) * 255
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    return image


def auto_orient(image: CvImage) -> CvImage:
    """
    Auto-orient image using Tesseract OSD (Orientation and Script Detection)
    Falls back to simple heuristic if Tesseract not available
    """
    try:
        import pytesseract
        osd = pytesseract.image_to_osd(image)
        
        # Parse orientation
        import re
        angle_match = re.search(r'Rotate: (\d+)', osd)
        if angle_match:
            angle = int(angle_match.group(1))
            if angle in [90, 180, 270]:
                return rotate_image(image, -angle)
    except Exception:
        pass
    
    # Fallback: try to detect using text orientation
    # This is a simplified heuristic
    return image


def image_to_base64(image: CvImage, format: str = "JPEG", quality: int = 90) -> str:
    """Convert OpenCV image to base64 string"""
    pil_img = cv_to_pil(image)
    buffer = BytesIO()
    pil_img.save(buffer, format=format, quality=quality)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def base64_to_image(base64_str: str) -> CvImage:
    """Convert base64 string to OpenCV image"""
    img_data = base64.b64decode(base64_str)
    return load_image(img_data)


def preprocess_pipeline(
    image: CvImage,
    config: dict
) -> CvImage:
    """
    Apply full preprocessing pipeline based on config
    
    Config keys:
        - auto_rotate: bool
        - deskew: bool
        - denoise: bool
        - denoise_method: str
        - enhance_contrast: bool
        - clahe_clip_limit: float
        - clahe_tile_grid: list
        - binarize: bool
        - binarize_method: str
        - max_dimension: int
        - min_dimension: int
    """
    # Resize first
    if config.get("max_dimension") or config.get("min_dimension"):
        image = resize_image(
            image,
            max_dimension=config.get("max_dimension", 3200),
            min_dimension=config.get("min_dimension", 640)
        )
    
    # Auto-rotate
    if config.get("auto_rotate", True):
        image = auto_orient(image)
    
    # Deskew
    if config.get("deskew", True):
        image, _ = deskew_image(image)
    
    # Denoise
    if config.get("denoise", True):
        image = denoise_image(
            image,
            method=config.get("denoise_method", "nlm"),
            strength=10.0
        )
    
    # Enhance contrast
    if config.get("enhance_contrast", True):
        image = enhance_contrast(
            image,
            method="clahe",
            clip_limit=config.get("clahe_clip_limit", 2.0),
            tile_grid=tuple(config.get("clahe_tile_grid", [8, 8]))
        )
    
    # Binarize
    if config.get("binarize", False):
        image = binarize_image(
            image,
            method=config.get("binarize_method", "sauvola")
        )
    
    return image


def extract_crops(image: CvImage, regions: List[dict], padding: int = 5) -> List[CvImage]:
    """Extract cropped regions from image"""
    crops = []
    for region in regions:
        bbox = region.get("bbox") or region.get("polygon")
        if bbox:
            if "polygon" in region:
                # Convert polygon to bbox
                pts = np.array(bbox, dtype=np.int32)
                x1, y1 = pts[:, 0].min(), pts[:, 1].min()
                x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            else:
                x1, y1, x2, y2 = bbox
            
            # Add padding
            h, w = image.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
    
    return crops