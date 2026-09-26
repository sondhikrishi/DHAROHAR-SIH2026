"""
Language Identification module
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
import re
from collections import Counter

from ..data_models import LanguageCode, ScriptType

logger = logging.getLogger(__name__)


# Unicode script ranges for Indic languages
SCRIPT_RANGES = {
    ScriptType.DEVANAGARI: [(0x0900, 0x097F)],
    ScriptType.BENGALI: [(0x0980, 0x09FF)],
    ScriptType.ORIYA: [(0x0B00, 0x0B7F)],
    ScriptType.TAMIL: [(0x0B80, 0x0BFF)],
    ScriptType.TELUGU: [(0x0C00, 0x0C7F)],
    ScriptType.KANNADA: [(0x0C80, 0x0CFF)],
    ScriptType.MALAYALAM: [(0x0D00, 0x0D7F)],
    ScriptType.GUJARATI: [(0x0A80, 0x0AFF)],
    ScriptType.GURMUKHI: [(0x0A00, 0x0A7F)],
    ScriptType.ARABIC: [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    ScriptType.LATIN: [(0x0000, 0x007F), (0x0080, 0x00FF), (0x0100, 0x017F), (0x0180, 0x024F)],
}

# Language to script mapping
LANGUAGE_SCRIPT = {
    LanguageCode.HI: ScriptType.DEVANAGARI,
    LanguageCode.MR: ScriptType.DEVANAGARI,
    LanguageCode.NE: ScriptType.DEVANAGARI,
    LanguageCode.BN: ScriptType.BENGALI,
    LanguageCode.AS: ScriptType.BENGALI,
    LanguageCode.OR: ScriptType.ORIYA,
    LanguageCode.TA: ScriptType.TAMIL,
    LanguageCode.TE: ScriptType.TELUGU,
    LanguageCode.KN: ScriptType.KANNADA,
    LanguageCode.ML: ScriptType.MALAYALAM,
    LanguageCode.GU: ScriptType.GUJARATI,
    LanguageCode.PA: ScriptType.GURMUKHI,
    LanguageCode.UR: ScriptType.ARABIC,
    LanguageCode.EN: ScriptType.LATIN,
}

# Script to possible languages
SCRIPT_LANGUAGES = {
    ScriptType.DEVANAGARI: [LanguageCode.HI, LanguageCode.MR, LanguageCode.NE],
    ScriptType.BENGALI: [LanguageCode.BN, LanguageCode.AS],
    ScriptType.ORIYA: [LanguageCode.OR],
    ScriptType.TAMIL: [LanguageCode.TA],
    ScriptType.TELUGU: [LanguageCode.TE],
    ScriptType.KANNADA: [LanguageCode.KN],
    ScriptType.MALAYALAM: [LanguageCode.ML],
    ScriptType.GUJARATI: [LanguageCode.GU],
    ScriptType.GURMUKHI: [LanguageCode.PA],
    ScriptType.ARABIC: [LanguageCode.UR],
    ScriptType.LATIN: [LanguageCode.EN],
}


class LanguageIdentifier:
    """Language identification for Indic text"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.method = config.get("method", "fasttext")
        self.fasttext_model = None
        self.min_confidence = config.get("min_confidence", 0.7)
        self.fallback_language = config.get("fallback_language", LanguageCode.HI)
        self.use_script_heuristic = config.get("use_script_heuristic", True)
        
        # Language-specific keywords for disambiguation
        self.language_keywords = {
            LanguageCode.HI: ["का", "की", "के", "में", "से", "को", "पर", "है", "हैं", "था", "थे"],
            LanguageCode.MR: ["चा", "ची", "चे", "मध्ये", "ते", "ला", "वर", "आहे", "होते"],
            LanguageCode.BN: ["ের", "এর", "দি", "তে", "কে", "এ", "আছে", "ছিল"],
            LanguageCode.TA: ["இன்", "அத்", "உள்ள", "என்", "ஆகும்", "இருந்தது"],
            LanguageCode.TE: ["లో", "కి", "నీ", "మograp", "ఉన్నాయి"],
            LanguageCode.KN: ["ಅಲ್ಲಿ", "ಗೆ", "ನಿರ್ವಹಣ", "ಆಗಿದೆ"],
            LanguageCode.ML: ["ൽ", "ക്ക്", "ന്റെ", "ഉണ്ട്"],
            LanguageCode.GU: ["નાં", "ને", "માં", "છે"],
            LanguageCode.PA: ["ਦੇ", "ਨੂੰ", "ਵਿੱਚ", "ਹੈ"],
            LanguageCode.UR: ["کا", "کی", "کے", "میں", "سے", "کو", "پر", "ہے"],
            LanguageCode.EN: ["the", "and", "of", "to", "in", "a", "is", "for", "on", "with"],
        }
    
    def initialize(self) -> bool:
        """Initialize language identification models"""
        if self.method == "fasttext":
            return self._init_fasttext()
        return True
    
    def _init_fasttext(self) -> bool:
        """Initialize fastText language identification model"""
        try:
            import fasttext
            import urllib.request
            import os
            
            model_path = self.config.get("fasttext_model", "models/lid.176.ftz")
            
            if not os.path.exists(model_path):
                logger.info("Downloading fastText language ID model...")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
                urllib.request.urlretrieve(url, model_path)
            
            self.fasttext_model = fasttext.load_model(model_path)
            logger.info("fastText language ID model loaded")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load fastText model: {e}. Using script heuristic only.")
            self.method = "heuristic"
            return True
    
    def identify(self, text: str) -> Tuple[LanguageCode, float]:
        """
        Identify language of text
        
        Returns:
            (language_code, confidence)
        """
        if not text or not text.strip():
            return self.fallback_language, 0.0
        
        # Try fastText first
        if self.method == "fasttext" and self.fasttext_model:
            lang, conf = self._identify_fasttext(text)
            if conf >= self.min_confidence:
                return lang, conf
        
        # Fallback to script heuristic
        if self.use_script_heuristic:
            lang, conf = self._identify_by_script(text)
            if conf >= self.min_confidence:
                return lang, conf
        
        # Final fallback: keyword matching
        lang, conf = self._identify_by_keywords(text)
        return lang, conf
    
    def _identify_fasttext(self, text: str) -> Tuple[LanguageCode, float]:
        """Identify using fastText model"""
        try:
            # fastText expects single line
            text = text.replace('\n', ' ')[:1000]
            predictions = self.fasttext_model.predict(text, k=3)
            
            # Parse labels (__label__hi, __label__en, etc.)
            for label, conf in zip(predictions[0], predictions[1]):
                lang_code = label.replace("__label__", "")
                try:
                    lang = LanguageCode(lang_code)
                    return lang, float(conf)
                except ValueError:
                    continue
            
        except Exception as e:
            logger.warning(f"fastText identification failed: {e}")
        
        return self.fallback_language, 0.0
    
    def _identify_by_script(self, text: str) -> Tuple[LanguageCode, float]:
        """Identify language by script analysis"""
        script_counts = Counter()
        
        for char in text:
            code = ord(char)
            for script, ranges in SCRIPT_RANGES.items():
                for start, end in ranges:
                    if start <= code <= end:
                        script_counts[script] += 1
                        break
        
        if not script_counts:
            return self.fallback_language, 0.0
        
        # Get dominant script
        dominant_script, count = script_counts.most_common(1)[0]
        total = sum(script_counts.values())
        confidence = count / total
        
        # Map script to language
        possible_langs = SCRIPT_LANGUAGES.get(dominant_script, [self.fallback_language])
        
        if len(possible_langs) == 1:
            return possible_langs[0], confidence
        
        # Disambiguate using keywords
        return self._disambiguate_by_keywords(text, possible_langs, confidence)
    
    def _disambiguate_by_keywords(
        self, 
        text: str, 
        candidates: List[LanguageCode], 
        base_confidence: float
    ) -> Tuple[LanguageCode, float]:
        """Disambiguate between languages sharing same script"""
        scores = {}
        
        for lang in candidates:
            keywords = self.language_keywords.get(lang, [])
            if not keywords:
                scores[lang] = 0.0
                continue
            
            matches = sum(1 for kw in keywords if kw in text)
            scores[lang] = matches / len(keywords)
        
        if not scores or max(scores.values()) == 0:
            # Return most common language for script
            return candidates[0], base_confidence * 0.5
        
        best_lang = max(scores, key=scores.get)
        keyword_confidence = scores[best_lang]
        
        # Combine script confidence with keyword confidence
        combined_confidence = (base_confidence + keyword_confidence) / 2
        
        return best_lang, combined_confidence
    
    def _identify_by_keywords(self, text: str) -> Tuple[LanguageCode, float]:
        """Identify language by keyword matching (last resort)"""
        scores = {}
        
        for lang, keywords in self.language_keywords.items():
            if not keywords:
                continue
            matches = sum(1 for kw in keywords if kw in text.lower())
            scores[lang] = matches / len(keywords)
        
        if not scores or max(scores.values()) == 0:
            return self.fallback_language, 0.1
        
        best_lang = max(scores, key=scores.get)
        return best_lang, scores[best_lang] * 0.5  # Low confidence
    
    def identify_batch(self, texts: List[str]) -> List[Tuple[LanguageCode, float]]:
        """Identify language for multiple texts"""
        return [self.identify(text) for text in texts]
    
    def get_script(self, text: str) -> ScriptType:
        """Get dominant script of text"""
        script_counts = Counter()
        
        for char in text:
            code = ord(char)
            for script, ranges in SCRIPT_RANGES.items():
                for start, end in ranges:
                    if start <= code <= end:
                        script_counts[script] += 1
                        break
        
        if not script_counts:
            return ScriptType.UNKNOWN
        
        return script_counts.most_common(1)[0][0]


def detect_script(text: str) -> ScriptType:
    """Standalone function to detect script"""
    identifier = LanguageIdentifier({})
    return identifier.get_script(text)


def get_script_name(script: ScriptType) -> str:
    """Get human-readable script name"""
    return script.value