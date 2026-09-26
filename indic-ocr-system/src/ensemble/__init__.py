"""
Ensemble voting module for combining multiple model predictions
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import logging

from ..data_models import TextRegion, BoundingBox, ModelPrediction, EnsembleResult

logger = logging.getLogger(__name__)


class EnsembleVoter:
    """Combine predictions from multiple models using various strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategy = config.get("strategy", "confidence_weighted")
        self.min_confidence = config.get("min_confidence", 0.55)
        self.iou_threshold = config.get("iou_threshold", 0.5)
        self.model_weights = config.get("model_weights", {
            "paddle_ocr": 1.0,
            "trocr": 0.9,
            "tesseract": 0.7,
            "easyocr": 0.8
        })
        self.language_overrides = config.get("language_overrides", {})
    
    def vote(
        self, 
        predictions: List[ModelPrediction],
        image_shape: Tuple[int, int],
        detected_language: Optional[str] = None
    ) -> EnsembleResult:
        """
        Combine predictions from multiple models
        
        Args:
            predictions: List of ModelPrediction from each model
            image_shape: (height, width) of original image
            detected_language: Optional language code for language-specific routing
            
        Returns:
            EnsembleResult with final regions
        """
        if not predictions:
            return EnsembleResult([], [], {}, 0.0)
        
        # Collect all regions from all models
        all_regions = []
        for pred in predictions:
            for region in pred.regions:
                region.metadata["source_model"] = pred.model_name
                all_regions.append(region)
        
        if not all_regions:
            return EnsembleResult([], predictions, {}, 0.0)
        
        # Group overlapping regions (same text location)
        groups = self._group_overlapping_regions(all_regions)
        
        # Apply voting strategy per group
        final_regions = []
        voting_details = {
            "num_groups": len(groups),
            "strategy": self.strategy,
            "groups": []
        }
        
        for group in groups:
            voted_region = self._vote_group(group, detected_language)
            if voted_region and voted_region.confidence >= self.min_confidence:
                final_regions.append(voted_region)
                voting_details["groups"].append({
                    "bbox": voted_region.bbox.to_list(),
                    "text": voted_region.text,
                    "confidence": voted_region.confidence,
                    "model_votes": {
                        r.metadata.get("source_model", "unknown"): {
                            "text": r.text,
                            "confidence": r.confidence
                        }
                        for r in group
                    }
                })
        
        # Sort final regions
        final_regions = self._sort_regions(final_regions)
        
        return EnsembleResult(
            final_regions=final_regions,
            model_predictions=predictions,
            voting_details=voting_details,
            processing_time=sum(p.metadata.get("total_time", 0) for p in predictions)
        )
    
    def _group_overlapping_regions(self, regions: List[TextRegion]) -> List[List[TextRegion]]:
        """Group regions that overlap significantly (IoU > threshold)"""
        used = set()
        groups = []
        
        for i, region in enumerate(regions):
            if i in used:
                continue
            
            group = [region]
            used.add(i)
            
            for j, other in enumerate(regions):
                if j in used or i == j:
                    continue
                
                iou = region.bbox.iou(other.bbox)
                if iou >= self.iou_threshold:
                    group.append(other)
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _vote_group(
        self, 
        group: List[TextRegion], 
        detected_language: Optional[str] = None
    ) -> Optional[TextRegion]:
        """Apply voting strategy to a group of overlapping regions"""
        if not group:
            return None
        
        if len(group) == 1:
            return group[0]
        
        if self.strategy == "confidence_weighted":
            return self._confidence_weighted_vote(group, detected_language)
        elif self.strategy == "weighted_average":
            return self._weighted_average_vote(group, detected_language)
        elif self.strategy == "majority_vote":
            return self._majority_vote(group)
        elif self.strategy == "best_single":
            return self._best_single_vote(group, detected_language)
        else:
            return self._confidence_weighted_vote(group, detected_language)
    
    def _confidence_weighted_vote(
        self, 
        group: List[TextRegion], 
        detected_language: Optional[str] = None
    ) -> TextRegion:
        """Weight votes by confidence * model_weight"""
        scores = defaultdict(float)
        texts = {}
        
        for region in group:
            model = region.metadata.get("source_model", "unknown")
            weight = self.model_weights.get(model, 0.5)
            
            # Language-specific weight adjustment
            if detected_language and detected_language in self.language_overrides:
                override = self.language_overrides[detected_language]
                if model == override.get("primary"):
                    weight *= 1.2
                elif model == override.get("secondary"):
                    weight *= 1.1
            
            score = region.confidence * weight
            text = region.text.strip()
            
            if text:
                scores[text] += score
                if text not in texts:
                    texts[text] = region
        
        if not scores:
            # Return highest confidence region
            return max(group, key=lambda r: r.confidence)
        
        # Get best text
        best_text = max(scores, key=scores.get)
        best_region = texts[best_text]
        
        # Create merged region
        merged = TextRegion(
            text=best_text,
            confidence=min(scores[best_text] / sum(self.model_weights.values()), 1.0),
            bbox=self._merge_bboxes([r.bbox for r in group]),
            language=best_region.language,
            script=best_region.script,
            model_source="ensemble",
            metadata={
                "source_models": [r.metadata.get("source_model") for r in group],
                "vote_scores": dict(scores),
                "strategy": "confidence_weighted"
            }
        )
        
        return merged
    
    def _weighted_average_vote(
        self, 
        group: List[TextRegion], 
        detected_language: Optional[str] = None
    ) -> TextRegion:
        """Average confidences weighted by model weights"""
        # Similar to confidence_weighted but different aggregation
        return self._confidence_weighted_vote(group, detected_language)
    
    def _majority_vote(self, group: List[TextRegion]) -> TextRegion:
        """Simple majority vote on text content"""
        text_counts = defaultdict(int)
        text_regions = {}
        
        for region in group:
            text = region.text.strip()
            if text:
                text_counts[text] += 1
                if text not in text_regions:
                    text_regions[text] = region
        
        if not text_counts:
            return max(group, key=lambda r: r.confidence)
        
        best_text = max(text_counts, key=text_counts.get)
        best_region = text_regions[best_text]
        
        # Confidence = majority ratio * avg confidence
        majority_ratio = text_counts[best_text] / len(group)
        avg_conf = np.mean([r.confidence for r in group if r.text.strip() == best_text])
        
        return TextRegion(
            text=best_text,
            confidence=majority_ratio * avg_conf,
            bbox=self._merge_bboxes([r.bbox for r in group]),
            language=best_region.language,
            script=best_region.script,
            model_source="ensemble",
            metadata={"strategy": "majority_vote", "vote_counts": dict(text_counts)}
        )
    
    def _best_single_vote(
        self, 
        group: List[TextRegion], 
        detected_language: Optional[str] = None
    ) -> TextRegion:
        """Pick single best model based on language routing"""
        if not detected_language or detected_language not in self.language_overrides:
            # Default: highest weighted confidence
            return max(group, key=lambda r: r.confidence * self.model_weights.get(
                r.metadata.get("source_model", "unknown"), 0.5
            ))
        
        override = self.language_overrides[detected_language]
        primary = override.get("primary")
        secondary = override.get("secondary")
        
        # Try primary model
        for region in group:
            if region.metadata.get("source_model") == primary:
                return region
        
        # Try secondary
        for region in group:
            if region.metadata.get("source_model") == secondary:
                return region
        
        # Fallback
        return max(group, key=lambda r: r.confidence)
    
    def _merge_bboxes(self, bboxes: List[BoundingBox]) -> BoundingBox:
        """Merge multiple bboxes into one enclosing bbox"""
        x1 = min(b.x1 for b in bboxes)
        y1 = min(b.y1 for b in bboxes)
        x2 = max(b.x2 for b in bboxes)
        y2 = max(b.y2 for b in bboxes)
        return BoundingBox(x1, y1, x2, y2)
    
    def _sort_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Sort regions top-to-bottom, left-to-right"""
        return sorted(regions, key=lambda r: (r.bbox.y1, r.bbox.x1))


def merge_predictions(
    predictions: List[ModelPrediction],
    strategy: str = "confidence_weighted",
    **kwargs
) -> EnsembleResult:
    """Convenience function to merge predictions"""
    voter = EnsembleVoter({
        "strategy": strategy,
        **kwargs
    })
    
    # Get image shape from first prediction
    image_shape = (1000, 1000)  # Default
    if predictions and predictions[0].regions:
        # Estimate from regions
        max_x = max(r.bbox.x2 for r in predictions[0].regions)
        max_y = max(r.bbox.y2 for r in predictions[0].regions)
        image_shape = (int(max_y) + 100, int(max_x) + 100)
    
    return voter.vote(predictions, image_shape)