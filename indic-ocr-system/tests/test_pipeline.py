"""
Integration tests for IndicOCR pipeline
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.indic_ocr import IndicOCR, create_ocr, PipelineConfig
from src.data_models import LanguageCode, TextRegion, BoundingBox
from src.detectors import create_detector, PaddleDetector
from src.recognizers import create_recognizer, PaddleRecognizer
from src.language_id import LanguageIdentifier
from src.postprocess import IndicCorrector, LandRecordParser
from src.ensemble import EnsembleVoter


class TestPipelineConfig:
    """Test pipeline configuration"""
    
    def test_default_config(self):
        config = PipelineConfig()
        assert config.detector_type == "paddle"
        assert "paddle_ocr" in config.recognizers
        assert "trocr" in config.recognizers
        assert config.device in ["auto", "cuda", "cpu", "mps"]
        assert len(config.languages) > 0
    
    def test_custom_config(self):
        config = PipelineConfig(
            languages=["hi", "en"],
            device="cpu",
            recognizers=["paddle_ocr"]
        )
        assert config.languages == ["hi", "en"]
        assert config.device == "cpu"
        assert config.recognizers == ["paddle_ocr"]


class TestLanguageIdentifier:
    """Test language identification"""
    
    def test_identify_hindi(self):
        identifier = LanguageIdentifier({})
        identifier.initialize()
        
        text = "यह एक हिंदी वाक्य है। खसरा नंबर 142/2A।"
        lang, conf = identifier.identify(text)
        assert lang == LanguageCode.HI
        assert conf > 0.5
    
    def test_identify_english(self):
        identifier = LanguageIdentifier({})
        identifier.initialize()
        
        text = "This is an English sentence. Survey number 142/2A."
        lang, conf = identifier.identify(text)
        assert lang == LanguageCode.EN
        assert conf > 0.5
    
    def test_identify_bengali(self):
        identifier = LanguageIdentifier({})
        identifier.initialize()
        
        text = "এটি একটি বাঙ্গলা বাক্য। দাগ নম্বর ১৪২/২A।"
        lang, conf = identifier.identify(text)
        assert lang == LanguageCode.BN
    
    def test_identify_tamil(self):
        identifier = LanguageIdentifier({})
        identifier.initialize()
        
        text = "இது ஒரு தமிழ் வாக்கியம்। சர்வே எண் 142/2A।"
        lang, conf = identifier.identify(text)
        assert lang == LanguageCode.TA
    
    def test_identify_mixed_script(self):
        identifier = LanguageIdentifier({})
        identifier.initialize()
        
        text = "Survey number 142/2A खसरा नंबर"
        lang, conf = identifier.identify(text)
        # Should detect one of the languages
        assert lang in [LanguageCode.HI, LanguageCode.EN]
    
    def test_identify_empty(self):
        identifier = LanguageIdentifier({})
        identifier.initialize()
        
        lang, conf = identifier.identify("")
        assert conf == 0.0
    
    def test_script_detection(self):
        identifier = LanguageIdentifier({})
        
        assert identifier.get_script("हिंदी") == "Devanagari"
        assert identifier.get_script("বাংলা") == "Bengali"
        assert identifier.get_script("தமிழ்") == "Tamil"
        assert identifier.get_script("English") == "Latin"
        assert identifier.get_script("123") == "Latin"  # Digits


class TestIndicCorrector:
    """Test Indic text correction"""
    
    def test_rule_based_correction_hindi(self):
        corrector = IndicCorrector({"enable_indic_correction": False})
        
        # Test digit normalization
        text = "खसरा १४२/२A"
        corrected = corrector._rule_based_correction(text, LanguageCode.HI)
        assert "142" in corrected
    
    def test_vocabulary_correction(self):
        corrector = IndicCorrector({
            "enable_indic_correction": False,
            "vocabularies": {
                "hi": ["खसरा", "खतौनी", "मौजा", "तहसील"]
            }
        })
        
        text = "खसर खतौन मौज"
        corrected = corrector._vocabulary_correction(text, LanguageCode.HI)
        # Should correct to closest vocabulary words
        assert "खसरा" in corrected or "खसर" in corrected
    
    def test_clean_person_name(self):
        from src.postprocess import LandRecordParser
        
        # Create dummy parser for testing
        parser = LandRecordParser.__new__(LandRecordParser)
        
        # Test Hindi titles removal
        name = "श्री राम कुमार शर्मा"
        cleaned = parser._clean_person_name(name)
        assert "श्री" not in cleaned
        
        # Test English titles
        name = "Mr. John Smith"
        cleaned = parser._clean_person_name(name)
        assert cleaned == "John Smith"


class TestLandRecordParser:
    """Test land record field extraction"""
    
    def test_parse_survey_number_hindi(self):
        # Create parser with minimal schema
        import tempfile
        import yaml
        
        schema = {
            "fields": {
                "survey_number": {
                    "patterns": [
                        {
                            "regex": "(?:खसरा|खसरा नंबर|khasra)[\\s:]*([A-Z0-9/\\-]+)",
                            "group": 1,
                            "languages": ["hi", "en"]
                        }
                    ]
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
        finally:
            Path(schema_path).unlink()
    
    def test_parse_area(self):
        from src.postprocess import LandRecordParser
        parser = LandRecordParser.__new__(LandRecordParser)
        
        area = "0.2520 हेक्टेयर"
        parsed = parser._parse_area(area)
        assert "0.2520" in parsed
        assert "हेक्टेयर" in parsed
    
    def test_normalize_date(self):
        from src.postprocess import LandRecordParser
        parser = LandRecordParser.__new__(LandRecordParser)
        
        # Test DD/MM/YYYY
        date = parser._normalize_date("15/03/2024")
        assert date == "2024-03-15"
        
        # Test DD/MM/YY
        date = parser._normalize_date("15/03/24")
        assert date == "2024-03-15"
    
    def test_parse_currency(self):
        from src.postprocess import LandRecordParser
        parser = LandRecordParser.__new__(LandRecordParser)
        
        currency = parser._parse_currency("₹ 50,000")
        assert "50000" in currency or "50,000" in currency
        assert "₹" in currency


class TestEnsembleVoter:
    """Test ensemble voting"""
    
    def test_confidence_weighted_vote(self):
        voter = EnsembleVoter({
            "strategy": "confidence_weighted",
            "model_weights": {
                "model_a": 1.0,
                "model_b": 0.8
            }
        })
        
        from src.data_models import ModelPrediction
        
        bbox = BoundingBox(10, 10, 100, 50)
        
        pred1 = ModelPrediction(
            model_name="model_a",
            regions=[
                TextRegion(text="Hello", confidence=0.9, bbox=bbox, model_source="model_a")
            ],
            processing_time=0.1
        )
        
        pred2 = ModelPrediction(
            model_name="model_b",
            regions=[
                TextRegion(text="Hello", confidence=0.8, bbox=bbox, model_source="model_b")
            ],
            processing_time=0.1
        )
        
        result = voter.vote([pred1, pred2], (200, 200))
        
        assert len(result.final_regions) == 1
        assert result.final_regions[0].text == "Hello"
        assert result.final_regions[0].confidence > 0.8
    
    def test_majority_vote(self):
        voter = EnsembleVoter({"strategy": "majority_vote"})
        
        from src.data_models import ModelPrediction
        
        bbox = BoundingBox(10, 10, 100, 50)
        
        # Two models agree, one disagrees
        preds = [
            ModelPrediction("m1", [TextRegion(text="Hello", confidence=0.9, bbox=bbox)], 0.1),
            ModelPrediction("m2", [TextRegion(text="Hello", confidence=0.8, bbox=bbox)], 0.1),
            ModelPrediction("m3", [TextRegion(text="Hallo", confidence=0.7, bbox=bbox)], 0.1),
        ]
        
        result = voter.vote(preds, (200, 200))
        
        assert len(result.final_regions) == 1
        assert result.final_regions[0].text == "Hello"
    
    def test_single_prediction(self):
        voter = EnsembleVoter({})
        
        from src.data_models import ModelPrediction
        
        bbox = BoundingBox(10, 10, 100, 50)
        pred = ModelPrediction(
            "m1", 
            [TextRegion(text="Hello", confidence=0.9, bbox=bbox)], 
            0.1
        )
        
        result = voter.vote([pred], (200, 200))
        
        assert len(result.final_regions) == 1
        assert result.final_regions[0].text == "Hello"


class TestIndicOCR:
    """Test main IndicOCR class (requires models)"""
    
    @pytest.mark.skip(reason="Requires model downloads")
    def test_create_ocr(self):
        ocr = create_ocr(languages=["hi", "en"], device="cpu")
        assert ocr is not None
        assert ocr.is_initialized
    
    @pytest.mark.skip(reason="Requires model downloads")
    def test_process_image(self):
        ocr = create_ocr(languages=["hi", "en"], device="cpu")
        
        # Create dummy image
        img = np.ones((200, 200, 3), dtype=np.uint8) * 255
        # Add some text-like patterns
        cv2.putText(img, "Test", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        result = ocr.process(img)
        assert isinstance(result.full_text, str)


# Fixtures
@pytest.fixture
def sample_image():
    """Create a sample test image"""
    img = np.ones((300, 400, 3), dtype=np.uint8) * 255
    # Add text-like rectangles
    cv2.rectangle(img, (50, 50), (200, 100), (0, 0, 0), -1)
    cv2.rectangle(img, (50, 150), (300, 200), (0, 0, 0), -1)
    return img


@pytest.fixture
def hindi_land_record_text():
    return """खसरा नंबर: 142/2A
खतौनी नंबर: 00345
मौजा: रामपुर
तहसील: सदर
जिला: लखनऊ
राज्य: उत्तर प्रदेश
रकबा: 0.2520 हेक्टेयर
भूमि प्रकार: कृषि भूमि
नामांतरण नंबर: MUT/2024/001234
पंजीकरण दिनांक: 15/03/2024"""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])