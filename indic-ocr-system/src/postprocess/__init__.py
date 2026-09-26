"""
Post-processing module for IndicOCR
"""

import re
import yaml
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from ..data_models import TextRegion, OCRResult, LanguageCode
from ..utils.metrics import normalize_text

logger = logging.getLogger(__name__)


# Common OCR error patterns for Indic languages
INDIC_OCR_CORRECTIONS = {
    "hi": {
        # Common confusions
        "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
        "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
        # Matra confusions
        "ि": "ी", "ु": "ू", "े": "ै", "ो": "ौ",
        # Common word corrections
        "खसरा": "खसरा", "खतौनी": "खतौनी", "मौजा": "मौजा",
        "तहसील": "तहसील", "जिला": "जिला", "रकबा": "रकबा",
    },
    "bn": {
        "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
        "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9",
    },
    "ta": {
        "௦": "0", "௧": "1", "௨": "2", "௩": "3", "௪": "4",
        "௫": "5", "௬": "6", "௭": "7", "௮": "8", "௯": "9",
    },
    "te": {
        "౦": "0", "౧": "1", "౨": "2", "౩": "3", "౪": "4",
        "౫": "5", "౬": "6", "౭": "7", "౮": "8", "౯": "9",
    },
    "kn": {
        "೦": "0", "೧": "1", "೨": "2", "೩": "3", "೪": "4",
        "೫": "5", "೬": "6", "೭": "7", "೮": "8", "೯": "9",
    },
    "ml": {
        "൦": "0", "൧": "1", "൨": "2", "൩": "3", "൪": "4",
        "൫": "5", "൬": "6", "൭": "7", "൮": "8", "൯": "9",
    },
    "gu": {
        "૦": "0", "૧": "1", "૨": "2", "૩": "3", "૪": "4",
        "૫": "5", "૬": "6", "૭": "7", "૮": "8", "૯": "9",
    },
    "pa": {
        "੦": "0", "੧": "1", "੨": "2", "੩": "3", "੪": "4",
        "੫": "5", "੬": "6", "੭": "7", "੮": "8", "੯": "9",
    },
    "or": {
        "୦": "0", "୧": "1", "୨": "2", "୩": "3", "୪": "4",
        "୫": "5", "୬": "6", "୭": "7", "୮": "8", "୯": "9",
    },
}

# Land record specific vocabulary for each language
LAND_RECORD_VOCAB = {
    "hi": [
        "खसरा", "खतौनी", "मौजा", "तहसील", "जिला", "राज्य", "रकबा", "भूमि",
        "कृषि", "आवासीय", "वाणिज्यिक", "नामांतरण", "दाखिल", "खारिज",
        "पंजीकरण", "रजिस्ट्री", "बैनामा", "विक्रेता", "क्रेता", "गवाह",
        "स्टाम्प", "शुल्क", "हेक्टेयर", "एकड़", "वर्ग", "मीटर", "गज"
    ],
    "bn": [
        "দাগ", "খতিয়ান", "মৌজা", "উপজেলা", "জেলা", "রকবা", "জমি",
        "কৃষি", "আবাসিক", "বাণিজ্যিক", "নামজারি", "দাখিল", "খারিজ",
        "নিবন্ধন", "বিক্রেতা", "ক্রেতা", "হেক্টের", "একর"
    ],
    "ta": [
        "தொடர்பு", "பட்டா", "கிராமம்", "தாலுகா", "மாவட்டம்", "நிலம்",
        "விவசாய", "அபிவாசி", "வணிக", "நாமமாற்று", "பதிவு",
        "விற்பனையாளர்", "வாங்குபவர்", "ஹெக்டேர்", "ஏக்கர்"
    ],
    "te": [
        "సర్వే", "పట్టు", "గ్రామం", "మండలం", "జిల్లా", "భూమి",
        "వ్యవసాయ", "ఆవాసిక", "వ్యాపారిక", "నామ మార్పు", "నివంధన",
        "విక్రేత", "క్రేత", "హెక్టేర్", "ఏకర"
    ],
    "kn": [
        "ಸರ್ವೇ", "ಪಟ್ಟೆ", "ಗ್ರಾಮ", "ತಾಲೂಕು", "ಜಿಲ್ಲೆ", "ಭೂಮಿ",
        "ಕೃಷಿ", "ನಿವಾಸ", "ವಾಣಿಜ್ಯ", "ನಾಮ ಬದಲಾವಣೆ", "ನೊಂದಣಿ",
        "ವಿಕ್ರೇತರ", "ಕ್ರೇತರ", "ಹೆಕ್ಟೇರ್", "ಏಕರ್"
    ],
    "ml": [
        "സർവേ", "പട്ടയം", "ഗ്രാമം", "താലൂക്ക്", "ജില്ല", "നിലം",
        "കൃഷി", "ആവാസിക", "വാണിജ്യ", "നാമ മാറ്റം", "നിബന്ധനം",
        "വിക്രേതാവ്", "വാങ്ങുന്നവർ", "ഹെക്ടെയർ", "ഏക്കർ"
    ],
    "gu": [
        "સર્વે", "સાતબારા", "ગામ", "તાલુકા", "જિલ્લો", "જમીન",
        "કૃષિ", "આવાસિક", "વ્યાપારિક", "નામ ફેરફાર", "નોંધણી",
        "વિક્રેતા", "ખરીદાર", "હેક્ટર", "એકર"
    ],
    "pa": [
        "ਸਰਵੇ", "ਜਮਾਬੰਦੀ", "ਗਾਂਵ", "ਤਹਿਸੀਲ", "ਜ਼ਿਲ੍ਹਾ", "ਜ਼ਮੀਨ",
        "ਖੇਤੀਬਾੜੀ", "ਆਵਾਸੀਕ", "ਵਿਆਪਾਰੀਕ", "ਨਾਮ ਬਦਲੀ", "ਰਜਿਸਟ੍ਰੀ",
        "ਵਿਕਰੇਤਾ", "ਖਰੀਦਦਾਰ", "ਹੈਕਟੇਅਰ", "ਏਕੜ"
    ],
    "ur": [
        "سرvey", "ختنون", "موعزہ", "تحصیل", "ضلع", "اراضی",
        "زرعی", "رہائشی", "تجارتی", "انتقال", "رجسٹری",
        "فروخت کنندہ", "خریدار", "ہیکٹر", "ایکر"
    ],
    "en": [
        "survey", "khata", "khasra", "village", "tehsil", "district",
        "state", "area", "hectare", "acre", "agricultural", "residential",
        "commercial", "mutation", "registration", "registry", "seller",
        "buyer", "witness", "stamp", "duty", "square", "meter"
    ]
}


class IndicCorrector:
    """Post-correction for Indic OCR output"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("correction_model", "ai4bharat/indicbert")
        self.model = None
        self.tokenizer = None
        self.enabled = config.get("enable_indic_correction", True)
        self.batch_size = config.get("correction_batch_size", 8)
        self.max_length = config.get("max_correction_length", 512)
        
        # Load vocabulary for languages
        self.vocabularies = config.get("vocabularies", LAND_RECORD_VOCAB)
    
    def initialize(self) -> bool:
        """Initialize correction model (IndicBERT)"""
        if not self.enabled:
            return True
        
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self.model.to(device)
            self.model.eval()
            self.device = device
            
            logger.info(f"IndicBERT corrector initialized on {device}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load IndicBERT: {e}. Using rule-based correction only.")
            self.enabled = False
            return True
    
    def correct(self, text: str, language: LanguageCode) -> str:
        """Apply corrections to text"""
        if not text:
            return text
        
        # Apply rule-based corrections first
        text = self._rule_based_correction(text, language)
        
        # Apply vocabulary-based correction
        text = self._vocabulary_correction(text, language)
        
        # Apply model-based correction (if available)
        if self.enabled and self.model:
            text = self._model_correction(text, language)
        
        return text
    
    def _rule_based_correction(self, text: str, language: LanguageCode) -> str:
        """Apply rule-based corrections"""
        lang_code = language.value
        
        # Normalize digits
        corrections = INDIC_OCR_CORRECTIONS.get(lang_code, {})
        for wrong, right in corrections.items():
            text = text.replace(wrong, right)
        
        # Fix common spacing issues
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _vocabulary_correction(self, text: str, language: LanguageCode) -> str:
        """Correct words using domain vocabulary"""
        lang_code = language.value
        vocab = self.vocabularies.get(lang_code, [])
        
        if not vocab:
            return text
        
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Clean word for matching
            clean_word = re.sub(r'[^\w]', '', word)
            
            if not clean_word:
                corrected_words.append(word)
                continue
            
            # Find closest vocabulary match
            best_match = self._find_closest_word(clean_word, vocab)
            
            if best_match and self._similarity(clean_word, best_match) > 0.7:
                # Preserve punctuation
                prefix = word[:len(word) - len(word.lstrip())]
                suffix = word[len(word.rstrip()):]
                corrected_words.append(prefix + best_match + suffix)
            else:
                corrected_words.append(word)
        
        return " ".join(corrected_words)
    
    def _find_closest_word(self, word: str, vocab: List[str]) -> Optional[str]:
        """Find closest vocabulary word using edit distance"""
        from ..utils.metrics import levenshtein_distance
        
        best_word = None
        best_dist = float('inf')
        
        for vocab_word in vocab:
            dist = levenshtein_distance(list(word), list(vocab_word))
            normalized_dist = dist / max(len(word), len(vocab_word))
            
            if normalized_dist < best_dist:
                best_dist = normalized_dist
                best_word = vocab_word
        
        return best_word if best_dist < 0.3 else None
    
    def _similarity(self, a: str, b: str) -> float:
        """Calculate string similarity"""
        from ..utils.metrics import levenshtein_distance
        if not a or not b:
            return 0.0
        dist = levenshtein_distance(list(a), list(b))
        return 1 - dist / max(len(a), len(b))
    
    def _model_correction(self, text: str, language: LanguageCode) -> str:
        """Apply IndicBERT-based correction"""
        try:
            import torch
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits.argmax(dim=-1)
            
            # Decode
            corrected = self.tokenizer.decode(predictions[0], skip_special_tokens=True)
            
            return corrected
            
        except Exception as e:
            logger.warning(f"Model correction failed: {e}")
            return text
    
    def correct_batch(self, texts: List[str], languages: List[LanguageCode]) -> List[str]:
        """Correct multiple texts"""
        return [self.correct(text, lang) for text, lang in zip(texts, languages)]


class LandRecordParser:
    """Parse structured fields from land record OCR text"""
    
    def __init__(self, schema_path: str):
        self.schema_path = schema_path
        self.schema = self._load_schema()
        self.field_patterns = self._compile_patterns()
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load field extraction schema from YAML"""
        try:
            with open(self.schema_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            return {"fields": {}}
    
    def _compile_patterns(self) -> Dict[str, List[Dict]]:
        """Compile regex patterns for each field"""
        patterns = {}
        
        for field_name, field_def in self.schema.get("fields", {}).items():
            patterns[field_name] = []
            
            for pattern_def in field_def.get("patterns", []):
                try:
                    regex = re.compile(pattern_def["regex"], re.IGNORECASE | re.UNICODE)
                    patterns[field_name].append({
                        "regex": regex,
                        "group": pattern_def.get("group", 1),
                        "languages": pattern_def.get("languages", []),
                        "validation": pattern_def.get("validation", {}),
                        "post_process": pattern_def.get("post_process"),
                        "confidence_boost": field_def.get("confidence_boost", 0.0)
                    })
                except Exception as e:
                    logger.warning(f"Failed to compile pattern for {field_name}: {e}")
        
        return patterns
    
    def parse(self, text: str, language: LanguageCode) -> Dict[str, Any]:
        """
        Extract structured fields from text
        
        Returns:
            Dict of field_name -> {value, confidence, bbox}
        """
        results = {}
        lang_code = language.value
        
        for field_name, patterns in self.field_patterns.items():
            best_match = None
            best_confidence = 0.0
            
            for pattern in patterns:
                # Check if pattern applies to this language
                if pattern["languages"] and lang_code not in pattern["languages"]:
                    continue
                
                matches = pattern["regex"].finditer(text)
                
                for match in matches:
                    try:
                        value = match.group(pattern["group"])
                        
                        # Validate
                        if not self._validate(value, pattern.get("validation", {})):
                            continue
                        
                        # Apply post-processing
                        if pattern.get("post_process"):
                            value = self._post_process(value, pattern["post_process"])
                        
                        # Calculate confidence
                        confidence = self._calculate_field_confidence(
                            value, field_name, lang_code
                        ) + pattern.get("confidence_boost", 0.0)
                        
                        if confidence > best_confidence:
                            best_confidence = min(confidence, 1.0)
                            best_match = value
                            
                    except IndexError:
                        continue
            
            if best_match:
                results[field_name] = {
                    "value": best_match,
                    "confidence": best_confidence
                }
        
        return results
    
    def _validate(self, value: str, validation: Dict) -> bool:
        """Validate extracted value"""
        if not value:
            return False
        
        # Pattern validation
        if "pattern" in validation:
            if not re.match(validation["pattern"], value):
                return False
        
        # Length validation
        if "min_length" in validation and len(value) < validation["min_length"]:
            return False
        if "max_length" in validation and len(value) > validation["max_length"]:
            return False
        
        return True
    
    def _post_process(self, value: str, processor: str) -> str:
        """Apply post-processing function"""
        if processor == "clean_person_name":
            return self._clean_person_name(value)
        elif processor == "parse_area":
            return self._parse_area(value)
        elif processor == "normalize_date":
            return self._normalize_date(value)
        elif processor == "parse_currency":
            return self._parse_currency(value)
        return value
    
    def _clean_person_name(self, name: str) -> str:
        """Clean person name"""
        # Remove titles
        titles = ["श्री", "श्रीमती", "सुश्री", "डॉ", "डॉक्टर", "Mr", "Mrs", "Ms", "Dr", "Shri", "Smt"]
        for title in titles:
            name = re.sub(rf'^{title}[\s.]+', '', name, flags=re.IGNORECASE)
        
        # Title case for Latin script
        if re.search(r'[A-Za-z]', name):
            name = name.title()
        
        return name.strip()
    
    def _parse_area(self, value: str) -> str:
        """Parse and standardize area"""
        # Extract number and unit
        match = re.search(r'([0-9,]+(?:\.[0-9]+)?)\s*(.+)', value)
        if match:
            num = match.group(1).replace(',', '')
            unit = match.group(2).strip()
            return f"{num} {unit}"
        return value
    
    def _normalize_date(self, value: str) -> str:
        """Normalize date to YYYY-MM-DD"""
        # Try multiple formats
        formats = [
            r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})',  # DD/MM/YYYY
            r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{2})',  # DD/MM/YY
        ]
        
        for fmt in formats:
            match = re.match(fmt, value.strip())
            if match:
                d, m, y = match.groups()
                if len(y) == 2:
                    y = "20" + y if int(y) < 50 else "19" + y
                return f"{y}-{int(m):02d}-{int(d):02d}"
        
        return value
    
    def _parse_currency(self, value: str) -> str:
        """Parse currency amount"""
        # Extract number
        match = re.search(r'([0-9,]+(?:\.[0-9]+)?)', value.replace(',', ''))
        if match:
            amount = match.group(1)
            # Detect currency
            if '₹' in value or 'Rs' in value or 'रु' in value:
                return f"₹{amount}"
            elif '৳' in value or 'টাকা' in value:
                return f"৳{amount}"
        return value
    
    def _calculate_field_confidence(self, value: str, field_name: str, lang_code: str) -> float:
        """Calculate confidence for extracted field"""
        base_confidence = 0.7
        
        # Boost for known vocabulary
        vocab = LAND_RECORD_VOCAB.get(lang_code, [])
        if any(v in value for v in vocab):
            base_confidence += 0.1
        
        # Boost for specific field types
        if field_name in ["survey_number", "khata_number", "khasra_number"]:
            # Should match alphanumeric pattern
            if re.match(r'^[A-Z0-9/\-]+$', value):
                base_confidence += 0.15
        elif field_name == "area":
            if re.search(r'\d+(?:\.\d+)?\s*(hectare|acre|हेक्टेयर|एकड़|sq)', value, re.IGNORECASE):
                base_confidence += 0.2
        elif field_name == "registration_date":
            if re.match(r'\d{4}-\d{2}-\d{2}', value):
                base_confidence += 0.2
        
        return min(base_confidence, 1.0)
    
    def parse_full_result(self, ocr_result: OCRResult) -> Dict[str, Any]:
        """Parse all fields from OCR result"""
        # Combine all text
        full_text = ocr_result.full_text
        language = ocr_result.detected_language
        
        # Parse fields
        fields = self.parse(full_text, language)
        
        # Add overall confidence
        field_confidences = [f["confidence"] for f in fields.values()]
        overall_confidence = sum(field_confidences) / len(field_confidences) if field_confidences else 0.0
        
        return {
            "fields": fields,
            "overall_confidence": overall_confidence,
            "language": language.value,
            "raw_text": full_text
        }


def clean_ocr_text(text: str, language: LanguageCode) -> str:
    """Standalone function to clean OCR text"""
    corrector = IndicCorrector({"enable_indic_correction": False})
    return corrector._rule_based_correction(text, language)


def extract_land_record_fields(text: str, language: LanguageCode, schema_path: str) -> Dict[str, Any]:
    """Standalone function to extract land record fields"""
    parser = LandRecordParser(schema_path)
    return parser.parse(text, language)