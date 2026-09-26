"""
Recognizer base class and implementations
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import logging
import time
from PIL import Image

from ..data_models import TextRegion, BoundingBox
from ..utils.image_utils import crop_image, cv_to_pil

logger = logging.getLogger(__name__)


class BaseRecognizer(ABC):
    """Abstract base class for text recognizers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.is_initialized = False
        self.supported_languages = config.get("languages", [])
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the recognizer model"""
        pass
    
    @abstractmethod
    def recognize(self, image: np.ndarray, regions: List[TextRegion]) -> List[TextRegion]:
        """
        Recognize text in given regions
        
        Args:
            image: Full image
            regions: List of TextRegions with bboxes but empty text
            
        Returns:
            List of TextRegions with recognized text and confidence
        """
        pass
    
    def crop_regions(self, image: np.ndarray, regions: List[TextRegion], padding: int = 3) -> List[np.ndarray]:
        """Crop image regions with padding"""
        crops = []
        h, w = image.shape[:2]
        
        for region in regions:
            x1 = max(0, int(region.bbox.x1) - padding)
            y1 = max(0, int(region.bbox.y1) - padding)
            x2 = min(w, int(region.bbox.x2) + padding)
            y2 = min(h, int(region.bbox.y2) + padding)
            
            if x2 > x1 and y2 > y1:
                crop = image[y1:y2, x1:x2]
                crops.append(crop)
            else:
                crops.append(None)
        
        return crops


class PaddleRecognizer(BaseRecognizer):
    """PaddleOCR text recognizer"""
    
    def initialize(self) -> bool:
        try:
            from paddleocr import PaddleOCR
            
            rec_model = self.config.get("rec_model", "PP-OCRv4_server_rec")
            lang = self.config.get("lang", "indian")
            use_gpu = self.config.get("use_gpu", True)
            use_angle_cls = self.config.get("use_angle_cls", True)
            
            self.model = PaddleOCR(
                use_angle_cls=use_angle_cls,
                use_gpu=use_gpu,
                rec_model_dir=rec_model,
                lang=lang,
                det=False,  # Don't load detector
                show_log=False
            )
            self.is_initialized = True
            logger.info(f"PaddleOCR recognizer initialized (lang={lang})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR recognizer: {e}")
            return False
    
    def recognize(self, image: np.ndarray, regions: List[TextRegion]) -> List[TextRegion]:
        if not self.is_initialized:
            if not self.initialize():
                return regions
        
        if not regions:
            return []
        
        try:
            crops = self.crop_regions(image, regions)
            
            results = []
            for i, (region, crop) in enumerate(zip(regions, crops)):
                if crop is None or crop.size == 0:
                    results.append(region)
                    continue
                
                start = time.time()
                try:
                    # PaddleOCR expects RGB
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    result = self.model.ocr(crop_rgb, det=False, rec=True, cls=False)
                    rec_time = time.time() - start
                    
                    if result and result[0]:
                        text, conf = result[0][0], result[0][1]
                        region.text = text
                        region.confidence = float(conf)
                        region.model_source = "paddle_ocr"
                        region.metadata["recognition_time"] = rec_time
                    else:
                        region.text = ""
                        region.confidence = 0.0
                        
                except Exception as e:
                    logger.warning(f"PaddleOCR recognition failed for region {i}: {e}")
                    region.text = ""
                    region.confidence = 0.0
                
                results.append(region)
            
            return results
            
        except Exception as e:
            logger.error(f"PaddleOCR recognition failed: {e}")
            return regions


class TrOCRRecognizer(BaseRecognizer):
    """TrOCR (Transformer-based) recognizer"""
    
    def initialize(self) -> bool:
        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            model_name = self.config.get("model_name", "microsoft/trocr-base-handwritten")
            fine_tuned_path = self.config.get("fine_tuned_path")
            
            device = self.config.get("device", "auto")
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.device = device
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            
            if fine_tuned_path:
                self.model = VisionEncoderDecoderModel.from_pretrained(fine_tuned_path)
            else:
                self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            self.model.to(device)
            self.model.eval()
            
            # Generation config
            self.max_length = self.config.get("max_length", 256)
            self.num_beams = self.config.get("num_beams", 4)
            self.early_stopping = self.config.get("early_stopping", True)
            
            self.is_initialized = True
            logger.info(f"TrOCR recognizer initialized on {device}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TrOCR recognizer: {e}")
            return False
    
    def recognize(self, image: np.ndarray, regions: List[TextRegion]) -> List[TextRegion]:
        if not self.is_initialized:
            if not self.initialize():
                return regions
        
        if not regions:
            return []
        
        try:
            import torch
            
            crops = self.crop_regions(image, regions)
            
            for i, (region, crop) in enumerate(zip(regions, crops)):
                if crop is None or crop.size == 0:
                    continue
                
                start = time.time()
                try:
                    # Convert to PIL
                    pil_img = cv_to_pil(crop)
                    
                    # Process
                    pixel_values = self.processor(images=pil_img, return_tensors="pt").pixel_values
                    pixel_values = pixel_values.to(self.device)
                    
                    # Generate
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            pixel_values,
                            max_length=self.max_length,
                            num_beams=self.num_beams,
                            early_stopping=self.early_stopping
                        )
                    
                    # Decode
                    text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    # TrOCR doesn't provide confidence directly
                    # Use heuristic based on length and special tokens
                    confidence = min(0.95, 0.5 + len(text) * 0.01)
                    
                    region.text = text
                    region.confidence = confidence
                    region.model_source = "trocr"
                    region.metadata["recognition_time"] = time.time() - start
                    
                except Exception as e:
                    logger.warning(f"TrOCR recognition failed for region {i}: {e}")
                    region.text = ""
                    region.confidence = 0.0
            
            return regions
            
        except Exception as e:
            logger.error(f"TrOCR recognition failed: {e}")
            return regions


class TesseractRecognizer(BaseRecognizer):
    """Tesseract OCR recognizer"""
    
    def initialize(self) -> bool:
        try:
            import pytesseract
            
            # Check tesseract availability
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
            
            self.tesseract_cmd = self.config.get("tesseract_cmd", "")
            if self.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            
            self.langs = self.config.get("langs", ["hin", "eng"])
            self.config_str = self.config.get("config", "--psm 6 --oem 3")
            
            self.is_initialized = True
            logger.info(f"Tesseract recognizer initialized (langs={'+'.join(self.langs)})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Tesseract recognizer: {e}")
            return False
    
    def recognize(self, image: np.ndarray, regions: List[TextRegion]) -> List[TextRegion]:
        if not self.is_initialized:
            if not self.initialize():
                return regions
        
        if not regions:
            return []
        
        try:
            import pytesseract
            
            crops = self.crop_regions(image, regions)
            
            for i, (region, crop) in enumerate(zip(regions, crops)):
                if crop is None or crop.size == 0:
                    continue
                
                start = time.time()
                try:
                    # Convert to PIL
                    pil_img = cv_to_pil(crop)
                    
                    # Get detailed output with confidence
                    data = pytesseract.image_to_data(
                        pil_img,
                        lang="+".join(self.langs),
                        config=self.config_str,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Combine all detected text
                    texts = []
                    confidences = []
                    
                    for j in range(len(data['text'])):
                        if data['text'][j].strip():
                            texts.append(data['text'][j])
                            confidences.append(data['conf'][j])
                    
                    if texts:
                        region.text = " ".join(texts)
                        # Average confidence (Tesseract uses 0-100)
                        region.confidence = np.mean(confidences) / 100.0
                    else:
                        region.text = ""
                        region.confidence = 0.0
                    
                    region.model_source = "tesseract"
                    region.metadata["recognition_time"] = time.time() - start
                    
                except Exception as e:
                    logger.warning(f"Tesseract recognition failed for region {i}: {e}")
                    region.text = ""
                    region.confidence = 0.0
            
            return regions
            
        except Exception as e:
            logger.error(f"Tesseract recognition failed: {e}")
            return regions


class EasyOCRRecognizer(BaseRecognizer):
    """EasyOCR recognizer"""
    
    def initialize(self) -> bool:
        try:
            import easyocr
            
            langs = self.config.get("langs", ["hi", "en"])
            gpu = self.config.get("gpu", True)
            model_storage = self.config.get("model_storage_dir", "models/easyocr")
            
            self.reader = easyocr.Reader(
                langs,
                gpu=gpu,
                model_storage_directory=model_storage,
                download_enabled=self.config.get("download_enabled", True)
            )
            
            self.is_initialized = True
            logger.info(f"EasyOCR recognizer initialized (langs={langs})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR recognizer: {e}")
            return False
    
    def recognize(self, image: np.ndarray, regions: List[TextRegion]) -> List[TextRegion]:
        if not self.is_initialized:
            if not self.initialize():
                return regions
        
        if not regions:
            return []
        
        try:
            crops = self.crop_regions(image, regions)
            
            for i, (region, crop) in enumerate(zip(regions, crops)):
                if crop is None or crop.size == 0:
                    continue
                
                start = time.time()
                try:
                    # EasyOCR works on full image, but we can pass crops
                    # For better accuracy, run on full image and match boxes
                    # Simplified: run on crop
                    result = self.reader.readtext(crop, detail=1)
                    
                    if result:
                        # Take highest confidence result
                        best = max(result, key=lambda x: x[2])
                        box, text, conf = best
                        
                        region.text = text
                        region.confidence = float(conf)
                    else:
                        region.text = ""
                        region.confidence = 0.0
                    
                    region.model_source = "easyocr"
                    region.metadata["recognition_time"] = time.time() - start
                    
                except Exception as e:
                    logger.warning(f"EasyOCR recognition failed for region {i}: {e}")
                    region.text = ""
                    region.confidence = 0.0
            
            return regions
            
        except Exception as e:
            logger.error(f"EasyOCR recognition failed: {e}")
            return regions


# Factory function
def create_recognizer(recognizer_type: str, config: Dict[str, Any]) -> BaseRecognizer:
    """Create recognizer instance by type"""
    recognizers = {
        "paddle": PaddleRecognizer,
        "paddle_ocr": PaddleRecognizer,
        "trocr": TrOCRRecognizer,
        "tesseract": TesseractRecognizer,
        "easyocr": EasyOCRRecognizer,
    }
    
    recognizer_class = recognizers.get(recognizer_type.lower())
    if not recognizer_class:
        raise ValueError(f"Unknown recognizer type: {recognizer_type}. Available: {list(recognizers.keys())}")
    
    return recognizer_class(config)