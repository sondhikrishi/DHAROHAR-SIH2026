"""
Core data models for IndicOCR
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum
import json


class LanguageCode(str, Enum):
    """Supported language codes"""
    HI = "hi"
    BN = "bn"
    TA = "ta"
    TE = "te"
    KN = "kn"
    ML = "ml"
    MR = "mr"
    GU = "gu"
    PA = "pa"
    OR = "or"
    AS = "as"
    UR = "ur"
    EN = "en"
    NE = "ne"
    AUTO = "auto"


class ScriptType(str, Enum):
    """Script types"""
    DEVANAGARI = "Devanagari"
    BENGALI = "Bengali"
    ORIYA = "Oriya"
    TAMIL = "Tamil"
    TELUGU = "Telugu"
    KANNADA = "Kannada"
    MALAYALAM = "Malayalam"
    GUJARATI = "Gujarati"
    GURMUKHI = "Gurmukhi"
    ARABIC = "Arabic"
    LATIN = "Latin"
    UNKNOWN = "Unknown"


@dataclass
class BoundingBox:
    """Bounding box for text region"""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def center(self) -> tuple:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]
    
    def to_polygon(self) -> List[List[float]]:
        return [[self.x1, self.y1], [self.x2, self.y1], [self.x2, self.y2], [self.x1, self.y2]]
    
    @classmethod
    def from_list(cls, coords: List[float]) -> "BoundingBox":
        if len(coords) == 4:
            return cls(coords[0], coords[1], coords[2], coords[3])
        elif len(coords) == 8:  # Polygon format
            xs = coords[0::2]
            ys = coords[1::2]
            return cls(min(xs), min(ys), max(xs), max(ys))
        raise ValueError(f"Invalid coordinates: {coords}")
    
    def iou(self, other: "BoundingBox") -> float:
        """Calculate IoU with another bounding box"""
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = self.width * self.height
        area2 = other.width * other.height
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


@dataclass
class TextRegion:
    """A detected text region with recognition results"""
    text: str
    confidence: float
    bbox: BoundingBox
    language: LanguageCode = LanguageCode.AUTO
    script: ScriptType = ScriptType.UNKNOWN
    model_source: str = "unknown"  # Which model produced this
    polygon: Optional[List[List[float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox.to_list(),
            "polygon": self.polygon,
            "language": self.language.value,
            "script": self.script.value,
            "model_source": self.model_source,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextRegion":
        return cls(
            text=data["text"],
            confidence=data["confidence"],
            bbox=BoundingBox.from_list(data["bbox"]),
            polygon=data.get("polygon"),
            language=LanguageCode(data.get("language", "auto")),
            script=ScriptType(data.get("script", "Unknown")),
            model_source=data.get("model_source", "unknown"),
            metadata=data.get("metadata", {})
        )


@dataclass
class OCRResult:
    """Complete OCR result for an image"""
    image_path: str
    regions: List[TextRegion]
    full_text: str
    detected_language: LanguageCode
    language_confidence: float
    processing_time: float
    model_used: str
    structured_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "regions": [r.to_dict() for r in self.regions],
            "full_text": self.full_text,
            "detected_language": self.detected_language.value,
            "language_confidence": self.language_confidence,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "structured_data": self.structured_data,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save_json(self, path: str) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OCRResult":
        return cls(
            image_path=data["image_path"],
            regions=[TextRegion.from_dict(r) for r in data["regions"]],
            full_text=data["full_text"],
            detected_language=LanguageCode(data["detected_language"]),
            language_confidence=data["language_confidence"],
            processing_time=data["processing_time"],
            model_used=data["model_used"],
            structured_data=data.get("structured_data"),
            metadata=data.get("metadata", {}),
            timestamp=data.get("timestamp", datetime.now().isoformat())
        )
    
    def get_text_by_language(self, lang: LanguageCode) -> str:
        """Get concatenated text for a specific language"""
        return " ".join(
            r.text for r in self.regions 
            if r.language == lang
        )
    
    def get_high_confidence_regions(self, threshold: float = 0.8) -> List[TextRegion]:
        """Filter regions by confidence threshold"""
        return [r for r in self.regions if r.confidence >= threshold]


@dataclass
class ModelPrediction:
    """Prediction from a single model"""
    model_name: str
    regions: List[TextRegion]
    processing_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleResult:
    """Result from ensemble voting"""
    final_regions: List[TextRegion]
    model_predictions: List[ModelPrediction]
    voting_details: Dict[str, Any]
    processing_time: float