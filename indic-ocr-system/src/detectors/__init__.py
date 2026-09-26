"""
Detector base class and implementations
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import logging
import time

from ..data_models import TextRegion, BoundingBox
from ..utils.image_utils import preprocess_pipeline

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """Abstract base class for text detectors"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.is_initialized = False

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the detector model."""
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[TextRegion]:
        """Detect text regions in an image."""
        pass

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image before detection."""
        return preprocess_pipeline(image, self.config)
    
    def filter_boxes(self, boxes: List[TextRegion], image_shape: Tuple[int, int]) -> List[TextRegion]:
        """Filter out invalid boxes"""
        h, w = image_shape[:2]
        filtered = []
        
        for box in boxes:
            # Check bounds
            if box.bbox.x1 < 0 or box.bbox.y1 < 0:
                continue
            if box.bbox.x2 > w or box.bbox.y2 > h:
                continue
            if box.bbox.width < 10 or box.bbox.height < 10:
                continue
            if box.confidence < self.config.get("min_confidence", 0.3):
                continue
            filtered.append(box)
        
        return filtered
    
    def sort_boxes(self, boxes: List[TextRegion]) -> List[TextRegion]:
        """Sort boxes top-to-bottom, left-to-right"""
        # Group by rows (similar y-coordinate)
        boxes = sorted(boxes, key=lambda b: b.bbox.y1)
        
        rows = []
        current_row = []
        row_threshold = 20  # pixels
        
        for box in boxes:
            if not current_row:
                current_row.append(box)
            else:
                # Check if same row
                avg_y = sum(b.bbox.y1 for b in current_row) / len(current_row)
                if abs(box.bbox.y1 - avg_y) < row_threshold:
                    current_row.append(box)
                else:
                    # Sort current row by x
                    current_row.sort(key=lambda b: b.bbox.x1)
                    rows.extend(current_row)
                    current_row = [box]
        
        if current_row:
            current_row.sort(key=lambda b: b.bbox.x1)
            rows.extend(current_row)
        
        return rows


class PaddleDetector(BaseDetector):
    """PaddleOCR text detector compatible with PaddleOCR 3.x"""

    def initialize(self) -> bool:
        try:
            import paddle
            # Disable oneDNN/MKLDNN on CPU.
            paddle.set_flags({"FLAGS_use_mkldnn": False})

            def initialize(self) -> bool:
    try:
        import paddle

        # MUST happen before importing PaddleOCR
        paddle.set_flags({"FLAGS_use_mkldnn": False})

        from paddleocr import PaddleOCR

        self.model = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        self.is_initialized = True
        logger.info("PaddleOCR detector initialized")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize PaddleOCR detector: {e}")
        return False

            # New PaddleOCR API
            self.model = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

            self.is_initialized = True
            logger.info("PaddleOCR detector initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR detector: {e}")
            return False

    def detect(self, image: np.ndarray) -> List[TextRegion]:
        if not self.is_initialized:
            if not self.initialize():
                return []

        try:
            processed = self.preprocess(image)

            start = time.time()

            import paddle
            paddle.set_flags({"FLAGS_use_mkldnn": False})

            results = self.model.predict(
                processed,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

            det_time = time.time() - start

            regions = []
            

            for result in results:
                # PaddleOCR 3.x returns a result object/dict-like object.
                data = result

                # Convert to dictionary when possible
                if hasattr(result, "json"):
                    try:
                        data = result.json
                        if callable(data):
                            data = data()
                    except Exception:
                        data = result

                if hasattr(result, "res"):
                    data = result.res

                if isinstance(data, dict):
                    boxes = (
                        data.get("dt_polys")
                        or data.get("rec_polys")
                        or data.get("textline_polys")
                    )

                    scores = data.get("rec_scores") or data.get(
                        "text_det_scores"
                    )

                    if boxes is None:
                        continue

                    for i, box in enumerate(boxes):
                        pts = np.array(box, dtype=np.float32)

                        if pts.ndim != 2 or pts.shape[0] < 4:
                            continue

                        x1 = float(pts[:, 0].min())
                        y1 = float(pts[:, 1].min())
                        x2 = float(pts[:, 0].max())
                        y2 = float(pts[:, 1].max())

                        confidence = 0.9

                        if scores is not None and i < len(scores):
                            try:
                                confidence = float(scores[i])
                            except Exception:
                                pass

                        bbox = BoundingBox(
                            x1,
                            y1,
                            x2,
                            y2
                        )

                        region = TextRegion(
                            text="",
                            confidence=confidence,
                            bbox=bbox,
                            polygon=pts.tolist(),
                            model_source="paddle_ocr",
                            metadata={
                                "detection_time": det_time
                            }
                        )

                        regions.append(region)

            regions = self.filter_boxes(
                regions,
                image.shape
            )

            regions = self.sort_boxes(regions)

            logger.info(
                f"PaddleOCR detected {len(regions)} text regions"
            )

            return regions

        except Exception as e:
            logger.error(
                f"PaddleOCR detection failed: {e}"
            )
            return []


class DBNetDetector(BaseDetector):
    """DBNet text detector (standalone)"""
    
    def initialize(self) -> bool:
        try:
            import torch
            from torchvision import transforms
            
            # This would load a custom DBNet model
            # For now, we'll use a placeholder
            logger.warning("DBNetDetector not fully implemented, using placeholder")
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DBNet detector: {e}")
            return False
    
    def detect(self, image: np.ndarray) -> List[TextRegion]:
        # Placeholder - would run DBNet inference
        logger.warning("DBNetDetector.detect() not implemented")
        return []


class CRAFTDetector(BaseDetector):
    """CRAFT text detector"""
    
    def initialize(self) -> bool:
        try:
            # Would load CRAFT model
            logger.warning("CRAFTDetector not fully implemented")
            self.is_initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize CRAFT detector: {e}")
            return False
    
    def detect(self, image: np.ndarray) -> List[TextRegion]:
        logger.warning("CRAFTDetector.detect() not implemented")
        return []


# Factory function
def create_detector(detector_type: str, config: Dict[str, Any]) -> BaseDetector:
    """Create detector instance by type"""
    detectors = {
        "paddle": PaddleDetector,
        "paddle_ocr": PaddleDetector,
        "dbnet": DBNetDetector,
        "craft": CRAFTDetector,
    }
    
    detector_class = detectors.get(detector_type.lower())
    if not detector_class:
        raise ValueError(f"Unknown detector type: {detector_type}. Available: {list(detectors.keys())}")
    
    return detector_class(config)