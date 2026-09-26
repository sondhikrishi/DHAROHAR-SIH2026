"""
Unit tests for IndicOCR utilities
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.image_utils import (
    load_image, resize_image, crop_image, rotate_image,
    deskew_image, denoise_image, enhance_contrast, binarize_image,
    preprocess_pipeline, cv_to_pil, pil_to_cv
)
from src.utils.metrics import (
    calculate_cer, calculate_wer, calculate_f1, 
    compute_metrics, levenshtein_distance, normalize_text
)
from src.data_models import TextRegion, BoundingBox, OCRResult, LanguageCode


class TestImageUtils:
    """Test image processing utilities"""
    
    def test_load_image_from_array(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        loaded = load_image(img)
        assert loaded.shape == (100, 100, 3)
        assert loaded.dtype == np.uint8
    
    def test_load_image_from_pil(self):
        from PIL import Image
        pil_img = Image.new("RGB", (100, 100), color="red")
        loaded = load_image(pil_img)
        assert loaded.shape == (100, 100, 3)
    
    def test_resize_image_downscale(self):
        img = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        resized = resize_image(img, max_dimension=1000)
        assert max(resized.shape[:2]) <= 1000
    
    def test_resize_image_upscale(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        resized = resize_image(img, min_dimension=200)
        assert min(resized.shape[:2]) >= 200
    
    def test_crop_image(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        cropped = crop_image(img, (10, 10, 50, 50))
        assert cropped.shape == (40, 40, 3)
    
    def test_crop_image_bounds(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Crop outside bounds should be clamped
        cropped = crop_image(img, (-10, -10, 150, 150))
        assert cropped.shape == (100, 100, 3)
    
    def test_rotate_image(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        rotated = rotate_image(img, 90)
        assert rotated.shape == (100, 100, 3)
    
    def test_denoise_image(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        denoised = denoise_image(img, method="gaussian")
        assert denoised.shape == img.shape
    
    def test_enhance_contrast(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        enhanced = enhance_contrast(img, method="clahe")
        assert enhanced.shape == img.shape
    
    def test_binarize_image(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        binary = binarize_image(img, method="otsu")
        assert binary.shape == img.shape
        # Should be binary (only 0 and 255)
        unique = np.unique(binary)
        assert len(unique) <= 2 or all(v in [0, 255] for v in unique)
    
    def test_preprocess_pipeline(self):
        img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        config = {
            "max_dimension": 300,
            "deskew": True,
            "denoise": True,
            "enhance_contrast": True,
            "binarize": False
        }
        processed = preprocess_pipeline(img, config)
        assert processed.shape[0] <= 300
        assert processed.shape[1] <= 300
    
    def test_cv_pil_conversion(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        pil_img = cv_to_pil(img)
        back = pil_to_cv(pil_img)
        assert back.shape == img.shape


class TestMetrics:
    """Test evaluation metrics"""
    
    def test_levenshtein_distance(self):
        assert levenshtein_distance([], []) == 0
        assert levenshtein_distance(['a'], []) == 1
        assert levenshtein_distance(['a', 'b'], ['a', 'b']) == 0
        assert levenshtein_distance(['a', 'b'], ['a', 'c']) == 1
        assert levenshtein_distance(['a', 'b', 'c'], ['a', 'c']) == 1
    
    def test_normalize_text(self):
        text = "  Hello   World  "
        normalized = normalize_text(text)
        assert normalized == "Hello World"
    
    def test_normalize_text_unicode(self):
        text = "Hello\u200bWorld"  # Zero-width space
        normalized = normalize_text(text)
        assert "\u200b" not in normalized
    
    def test_calculate_cer_exact(self):
        cer = calculate_cer("hello world", "hello world")
        assert cer == 0.0
    
    def test_calculate_cer_substitution(self):
        cer = calculate_cer("hello world", "hello word")
        assert 0 < cer < 0.2
    
    def test_calculate_cer_insertion(self):
        cer = calculate_cer("hello", "hello world")
        assert cer > 0
    
    def test_calculate_cer_deletion(self):
        cer = calculate_cer("hello world", "hello")
        assert cer > 0
    
    def test_calculate_cer_empty(self):
        cer = calculate_cer("", "")
        assert cer == 0.0
        cer = calculate_cer("hello", "")
        assert cer == 1.0
        cer = calculate_cer("", "hello")
        assert cer == 1.0
    
    def test_calculate_wer_exact(self):
        wer = calculate_wer("hello world", "hello world")
        assert wer == 0.0
    
    def test_calculate_wer_different_words(self):
        wer = calculate_wer("hello world", "hello there")
        assert wer == 0.5  # 1 substitution out of 2 words
    
    def test_calculate_f1_exact(self):
        f1 = calculate_f1("hello world", "hello world")
        assert f1 == 1.0
    
    def test_calculate_f1_partial(self):
        f1 = calculate_f1("hello world", "hello there")
        assert 0 < f1 < 1.0
    
    def test_compute_metrics(self):
        references = ["hello world", "test case", "another example"]
        hypotheses = ["hello world", "test case", "another exampl"]
        languages = ["en", "en", "en"]
        
        metrics = compute_metrics(references, hypotheses, languages)
        
        assert isinstance(metrics.cer, float)
        assert isinstance(metrics.wer, float)
        assert isinstance(metrics.f1, float)
        assert metrics.num_samples == 3
        assert len(metrics.per_sample) == 3
    
    def test_compute_metrics_by_language(self):
        references = ["hello", "namaste", "bonjour"]
        hypotheses = ["hello", "namaste", "bonjour"]
        languages = ["en", "hi", "fr"]
        
        metrics = compute_metrics(references, hypotheses, languages)
        assert metrics.num_samples == 3


class TestDataModels:
    """Test data models"""
    
    def test_bounding_box(self):
        bbox = BoundingBox(10, 20, 50, 60)
        assert bbox.width == 40
        assert bbox.height == 40
        assert bbox.center == (30.0, 40.0)
        assert bbox.to_list() == [10, 20, 50, 60]
    
    def test_bounding_box_iou(self):
        bbox1 = BoundingBox(0, 0, 100, 100)
        bbox2 = BoundingBox(50, 50, 150, 150)
        iou = bbox1.iou(bbox2)
        # Intersection: 50x50 = 2500, Union: 10000 + 10000 - 2500 = 17500
        assert abs(iou - 2500/17500) < 0.001
    
    def test_bounding_box_from_list(self):
        bbox = BoundingBox.from_list([10, 20, 50, 60])
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 50
        assert bbox.y2 == 60
    
    def test_text_region(self):
        bbox = BoundingBox(10, 20, 50, 60)
        region = TextRegion(
            text="Hello",
            confidence=0.95,
            bbox=bbox,
            language=LanguageCode.HI,
            model_source="test"
        )
        assert region.text == "Hello"
        assert region.confidence == 0.95
    
    def test_text_region_to_dict(self):
        bbox = BoundingBox(10, 20, 50, 60)
        region = TextRegion(
            text="Hello",
            confidence=0.95,
            bbox=bbox,
            language=LanguageCode.HI,
            model_source="test"
        )
        d = region.to_dict()
        assert d["text"] == "Hello"
        assert d["confidence"] == 0.95
        assert d["language"] == "hi"
    
    def test_ocr_result(self):
        bbox = BoundingBox(10, 20, 50, 60)
        region = TextRegion(
            text="Hello",
            confidence=0.95,
            bbox=bbox,
            language=LanguageCode.HI
        )
        result = OCRResult(
            image_path="test.jpg",
            regions=[region],
            full_text="Hello",
            detected_language=LanguageCode.HI,
            language_confidence=0.9,
            processing_time=1.5,
            model_used="test"
        )
        assert result.full_text == "Hello"
        assert len(result.regions) == 1
    
    def test_ocr_result_json(self):
        bbox = BoundingBox(10, 20, 50, 60)
        region = TextRegion(text="Hello", confidence=0.95, bbox=bbox)
        result = OCRResult(
            image_path="test.jpg",
            regions=[region],
            full_text="Hello",
            detected_language=LanguageCode.HI,
            language_confidence=0.9,
            processing_time=1.5,
            model_used="test"
        )
        json_str = result.to_json()
        assert "Hello" in json_str
        assert "hi" in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])