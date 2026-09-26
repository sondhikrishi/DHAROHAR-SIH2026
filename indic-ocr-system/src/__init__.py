"""
IndicOCR - Multilingual Indian Language OCR System
====================================================

A production-ready OCR system optimized for Indian land records
and documents, supporting 12+ Indian languages.

Author: SIH Hackathon Team
License: MIT
"""

__version__ = "1.0.0"
__author__ = "SIH Hackathon Team"
__email__ = "team@sih-hackathon.com"
__license__ = "MIT"

from .indic_ocr import IndicOCR, OCRResult, TextRegion

__all__ = [
    "IndicOCR",
    "OCRResult",
    "TextRegion",
]