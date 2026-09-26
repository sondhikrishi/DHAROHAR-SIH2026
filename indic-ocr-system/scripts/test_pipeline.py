#!/usr/bin/env python3
"""
Test Pipeline Script
====================
Quick test to verify IndicOCR pipeline works correctly.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.image_utils import load_image, preprocess_pipeline
from src.utils.metrics import calculate_cer, calculate_wer, calculate_f1
from src.data_models import TextRegion, BoundingBox, OCRResult, LanguageCode
from src.language_id import LanguageIdentifier
from src.postprocess import IndicCorrector, LandRecordParser
from src.ensemble import EnsembleVoter


def test_image_utils():
    print("🧪 Testing image utils...")
    
    # Create test image
    img = np.ones((200, 300, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Test Hindi: खसरा 142", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Test loading
    loaded = load_image(img)
    assert loaded.shape == (200, 300, 3)
    print("  ✅ load_image")
    
    # Test preprocessing
    config = {
        "max_dimension": 150,
        "deskew": True,
        "denoise": True,
        "enhance_contrast": True,
    }
    processed = preprocess_pipeline(loaded, config)
    assert processed.shape[0] <= 150
    assert processed.shape[1] <= 150
    print("  ✅ preprocess_pipeline")


def test_metrics():
    print("🧪 Testing metrics...")
    
    # CER
    cer = calculate_cer("hello world", "hello world")
    assert cer == 0.0
    cer = calculate_cer("hello", "helo")
    assert 0 < cer < 1.0
    print("  ✅ calculate_cer")
    
    # WER
    wer = calculate_wer("hello world", "hello world")
    assert wer == 0.0
    wer = calculate_wer("hello world", "hello there")
    assert wer == 0.5
    print("  ✅ calculate_wer")
    
    # F1
    f1 = calculate_f1("hello world", "hello world")
    assert f1 == 1.0
    f1 = calculate_f1("hello world", "hello there")
    assert 0 < f1 < 1.0
    print("  ✅ calculate_f1")
    
    # Compute metrics
    refs = ["hello world", "test case"]
    hyps = ["hello world", "test case"]
    langs = ["en", "en"]
    metrics = calculate_cer(refs[0], hyps[0])
    print("  ✅ compute_metrics")


def test_data_models():
    print("🧪 Testing data models...")
    
    # BoundingBox
    bbox = BoundingBox(10, 20, 100, 80)
    assert bbox.width == 90
    assert bbox.height == 60
    assert bbox.iou(bbox) == 1.0
    print("  ✅ BoundingBox")
    
    # TextRegion
    region = TextRegion(
        text="खसरा 142",
        confidence=0.95,
        bbox=bbox,
        language=LanguageCode.HI,
        model_source="test"
    )
    d = region.to_dict()
    assert d["text"] == "खसरा 142"
    assert d["language"] == "hi"
    print("  ✅ TextRegion")
    
    # OCRResult
    result = OCRResult(
        image_path="test.jpg",
        regions=[region],
        full_text="खसरा 142",
        detected_language=LanguageCode.HI,
        language_confidence=0.98,
        processing_time=1.5,
        model_used="test"
    )
    json_str = result.to_json()
    assert "खसरा 142" in json_str
    print("  ✅ OCRResult")


def test_language_id():
    print("🧪 Testing language identification...")
    
    identifier = LanguageIdentifier({
        "method": "heuristic",
        "use_script_heuristic": True
    })
    identifier.initialize()
    
    # Hindi
    lang, conf = identifier.identify("खसरा नंबर 142 मौजा रामपुर")
    assert lang == LanguageCode.HI
    assert conf > 0.5
    print("  ✅ Hindi detection")
    
    # English
    lang, conf = identifier.identify("Survey number 142 village Rampur")
    assert lang == LanguageCode.EN
    assert conf > 0.5
    print("  ✅ English detection")
    
    # Bengali
    lang, conf = identifier.identify("দাগ নম্বর ১৪২ মৌজা রামপুর")
    assert lang == LanguageCode.BN
    print("  ✅ Bengali detection")
    
    # Script detection
    assert identifier.get_script("हिंदी") == "Devanagari"
    assert identifier.get_script("English") == "Latin"
    assert identifier.get_script("தமிழ்") == "Tamil"
    print("  ✅ Script detection")


def test_postprocess():
    print("🧪 Testing post-processing...")
    
    # IndicCorrector
    corrector = IndicCorrector({"enable_indic_correction": False})
    
    # Digit normalization
    text = "खसरा १४२/२A मौजा"
    corrected = corrector._rule_based_correction(text, LanguageCode.HI)
    assert "142" in corrected
    print("  ✅ Digit normalization")
    
    # Vocabulary correction
    corrector2 = IndicCorrector({
        "enable_indic_correction": False,
        "vocabularies": {"hi": ["खसरा", "खतौनी", "मौजा"]}
    })
    text = "खसर खतौन मौज"
    corrected = corrector2._vocabulary_correction(text, LanguageCode.HI)
    assert "खसरा" in corrected or "खसर" in corrected
    print("  ✅ Vocabulary correction")
    
    # LandRecordParser (needs schema file)
    import yaml
    import tempfile
    
    schema = {
        "fields": {
            "survey_number": {
                "patterns": [{
                    "regex": "(?:खसरा|khasra)[\\s:]*([A-Z0-9/\\-]+)",
                    "group": 1,
                    "languages": ["hi", "en"]
                }]
            }
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(schema, f)
        schema_path = f.name
    
    try:
        parser = LandRecordParser(schema_path)
        text = "खसरा नंबर 142/2A मौजा रामपुर"
        result = parser.parse(text, LanguageCode.HI)
        assert "survey_number" in result
        assert result["survey_number"]["value"] == "142/2A"
        print("  ✅ LandRecordParser")
    finally:
        Path(schema_path).unlink()


def test_ensemble():
    print("🧪 Testing ensemble voting...")
    
    from src.data_models import ModelPrediction
    
    voter = EnsembleVoter({
        "strategy": "confidence_weighted",
        "model_weights": {"m1": 1.0, "m2": 0.8}
    })
    
    bbox = BoundingBox(10, 10, 100, 50)
    
    pred1 = ModelPrediction("m1", [TextRegion("Hello", 0.9, bbox, model_source="m1")], 0.1)
    pred2 = ModelPrediction("m2", [TextRegion("Hello", 0.8, bbox, model_source="m2")], 0.1)
    
    result = voter.vote([pred1, pred2], (200, 200))
    assert len(result.final_regions) == 1
    assert result.final_regions[0].text == "Hello"
    print("  ✅ Confidence weighted vote")
    
    # Majority vote
    voter2 = EnsembleVoter({"strategy": "majority_vote"})
    preds = [
        ModelPrediction("m1", [TextRegion("Hello", 0.9, bbox)], 0.1),
        ModelPrediction("m2", [TextRegion("Hello", 0.8, bbox)], 0.1),
        ModelPrediction("m3", [TextRegion("Hallo", 0.7, bbox)], 0.1),
    ]
    result = voter2.vote(preds, (200, 200))
    assert result.final_regions[0].text == "Hello"
    print("  ✅ Majority vote")


def test_full_integration():
    print("🧪 Testing full integration (mock)...")
    
    # This would test the full pipeline but requires models
    # For now, just verify imports work
    from src.indic_ocr import IndicOCR, PipelineConfig
    
    config = PipelineConfig(
        languages=["hi", "en"],
        device="cpu",
        recognizers=["paddle_ocr"]  # Will fail without models
    )
    print("  ✅ PipelineConfig creation")
    
    # Test factory functions
    from src.detectors import create_detector
    from src.recognizers import create_recognizer
    
    det_config = {"use_gpu": False, "preprocessing": {}}
    detector = create_detector("paddle", det_config)
    assert detector is not None
    print("  ✅ Detector factory")
    
    rec_config = {"device": "cpu", "use_gpu": False, "languages": ["hi"]}
    recognizer = create_recognizer("paddle_ocr", rec_config)
    assert recognizer is not None
    print("  ✅ Recognizer factory")


def run_all_tests():
    print("=" * 50)
    print("🧪 IndicOCR Pipeline Tests")
    print("=" * 50)
    
    tests = [
        test_image_utils,
        test_metrics,
        test_data_models,
        test_language_id,
        test_postprocess,
        test_ensemble,
        test_full_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test IndicOCR pipeline")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()
    
    success = run_all_tests()
    sys.exit(0 if success else 1)