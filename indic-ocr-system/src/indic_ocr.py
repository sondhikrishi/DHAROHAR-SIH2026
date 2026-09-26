"""
Main IndicOCR Pipeline - Production-ready multilingual OCR for Indian languages
"""

import cv2
import numpy as np
import time
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .data_models import (
    OCRResult, TextRegion, LanguageCode, ModelPrediction,
    BoundingBox
)
from .detectors import create_detector, BaseDetector
from .recognizers import create_recognizer, BaseRecognizer
from .language_id import LanguageIdentifier
from .postprocess import IndicCorrector, LandRecordParser, clean_ocr_text
from .ensemble import EnsembleVoter
from .utils.image_utils import (
    load_image, preprocess_pipeline, extract_crops,
    cv_to_pil, pil_to_cv
)
from .utils.visualization import create_visualization, save_visualization

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for IndicOCR pipeline"""
    # Model settings
    detector_type: str = "paddle"
    recognizers: List[str] = None  # List of recognizer types
    languages: List[str] = None
    
    # Device
    device: str = "auto"  # auto, cuda, cpu, mps
    
    # Preprocessing
    preprocessing: Dict[str, Any] = None
    
    # Language ID
    language_id: Dict[str, Any] = None
    
    # Ensemble
    ensemble: Dict[str, Any] = None
    
    # Post-processing
    postprocess: Dict[str, Any] = None
    
    # Output
    output: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.recognizers is None:
            self.recognizers = ["paddle_ocr", "trocr", "tesseract"]
        if self.languages is None:
            self.languages = ["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"]
        if self.preprocessing is None:
            self.preprocessing = {}
        if self.language_id is None:
            self.language_id = {}
        if self.ensemble is None:
            self.ensemble = {}
        if self.postprocess is None:
            self.postprocess = {}
        if self.output is None:
            self.output = {}


class IndicOCR:
    """
    Main IndicOCR Pipeline
    
    Features:
    - Multi-model ensemble (PaddleOCR, TrOCR, Tesseract, EasyOCR)
    - Automatic language detection
    - Indic language post-correction
    - Land record field extraction
    - Batch processing
    - Visualization
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
        **kwargs
    ):
        """
        Initialize IndicOCR
        
        Args:
            config_path: Path to YAML config file
            config: PipelineConfig object
            **kwargs: Override config values
        """
        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        elif config:
            self.config = config
        else:
            self.config = PipelineConfig()
        
        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Initialize components
        self.detector: Optional[BaseDetector] = None
        self.recognizers: Dict[str, BaseRecognizer] = {}
        self.language_id: Optional[LanguageIdentifier] = None
        self.corrector: Optional[IndicCorrector] = None
        self.land_parser: Optional[LandRecordParser] = None
        self.ensemble_voter: Optional[EnsembleVoter] = None
        
        self.is_initialized = False
        self._init_device()
    
    def _load_config(self, config_path: str) -> PipelineConfig:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        
        # Map YAML keys to PipelineConfig
        config = PipelineConfig()
        config.detector_type = cfg_dict.get("models", {}).get("paddle_ocr", {}).get("enabled", True) and "paddle" or "none"
        
        # Determine enabled recognizers
        recognizers = []
        models = cfg_dict.get("models", {})
        if models.get("paddle_ocr", {}).get("enabled"):
            recognizers.append("paddle_ocr")
        if models.get("trocr", {}).get("enabled"):
            recognizers.append("trocr")
        if models.get("tesseract", {}).get("enabled"):
            recognizers.append("tesseract")
        if models.get("easyocr", {}).get("enabled"):
            recognizers.append("easyocr")
        config.recognizers = recognizers
        
        config.device = cfg_dict.get("device", "auto")
        config.preprocessing = cfg_dict.get("preprocessing", {})
        config.language_id = cfg_dict.get("language_id", {})
        config.ensemble = cfg_dict.get("ensemble", {})
        config.postprocess = cfg_dict.get("postprocess", {})
        config.output = cfg_dict.get("output", {})
        
        return config
    
    def _init_device(self):
        """Initialize device (CPU/GPU)"""
        import torch
        
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device
        
        logger.info(f"Using device: {self.device}")
    
    def initialize(self) -> bool:
        """Initialize all models and components"""
        logger.info("Initializing IndicOCR pipeline...")
        
        try:
            # Initialize detector
            det_config = {
                "use_gpu": self.device == "cuda",
                "preprocessing": self.config.preprocessing,
                **self._get_detector_config()
            }
            self.detector = create_detector(self.config.detector_type, det_config)
            if not self.detector.initialize():
                logger.error("Detector initialization failed")
                return False
            
            # Initialize recognizers
            for rec_type in self.config.recognizers:
                rec_config = {
                    "device": self.device,
                    "use_gpu": self.device == "cuda",
                    "languages": self.config.languages,
                    **self._get_recognizer_config(rec_type)
                }
                recognizer = create_recognizer(rec_type, rec_config)
                if recognizer.initialize():
                    self.recognizers[rec_type] = recognizer
                    logger.info(f"Initialized recognizer: {rec_type}")
                else:
                    logger.warning(f"Failed to initialize recognizer: {rec_type}")
            
            if not self.recognizers:
                logger.error("No recognizers initialized")
                return False
            
            # Initialize language identification
            self.language_id = LanguageIdentifier(self.config.language_id)
            self.language_id.initialize()
            
            # Initialize post-processor
            self.corrector = IndicCorrector(self.config.postprocess)
            self.corrector.initialize()
            
            # Initialize land record parser
            schema_path = self.config.postprocess.get("land_record_schema", "configs/land_record_schema.yaml")
            self.land_parser = LandRecordParser(schema_path)
            
            # Initialize ensemble voter
            self.ensemble_voter = EnsembleVoter(self.config.ensemble)
            
            self.is_initialized = True
            logger.info("IndicOCR pipeline initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return False
    
    def _get_detector_config(self) -> Dict[str, Any]:
        """Get detector-specific config from YAML"""
        # This would be loaded from model_config.yaml
        return {
            "det_model": "PP-OCRv4_server_det",
            "use_angle_cls": True,
        }
    
    def _get_recognizer_config(self, rec_type: str) -> Dict[str, Any]:
        """Get recognizer-specific config from YAML"""
        configs = {
            "paddle_ocr": {
                "rec_model": "PP-OCRv4_server_rec",
                "lang": "indian",
                "use_angle_cls": True,
            },
            "trocr": {
                "model_name": "microsoft/trocr-base-handwritten",
                "fine_tuned_path": "models/trocr-indic-finetuned",
                "max_length": 256,
                "num_beams": 4,
            },
            "tesseract": {
                "langs": ["hin", "ben", "tel", "mar", "tam", "guj", "kan", "mal", "pan", "ori", "asm", "urd", "eng"],
                "config": "--psm 6 --oem 3",
            },
            "easyocr": {
                "langs": ["hi", "bn", "ta", "te", "kn", "ml", "en"],
                "gpu": self.device == "cuda",
            }
        }
        return configs.get(rec_type, {})
    
    def process(
        self,
        image: Union[str, np.ndarray, bytes],
        return_visualization: bool = False,
        return_crops: bool = False
    ) -> OCRResult:
        """
        Process a single image
        
        Args:
            image: Image path, numpy array, or bytes
            return_visualization: Whether to include visualization in result
            return_crops: Whether to include cropped regions
            
        Returns:
            OCRResult with recognized text and metadata
        """
        if not self.is_initialized:
            self.initialize()
        
        start_time = time.time()
        
        # Load image
        cv_image = load_image(image)
        original_image = cv_image.copy()
        
        # Step 1: Text Detection
        det_start = time.time()
        regions = self.detector.detect(cv_image)
        det_time = time.time() - det_start
        
        if not regions:
            logger.warning("No text regions detected")
            return OCRResult(
                image_path=str(image) if isinstance(image, str) else "array",
                regions=[],
                full_text="",
                detected_language=LanguageCode.HI,
                language_confidence=0.0,
                processing_time=time.time() - start_time,
                model_used="ensemble"
            )
        
        logger.debug(f"Detected {len(regions)} text regions in {det_time:.3f}s")
        
        # Step 2: Text Recognition (parallel across models)
        rec_start = time.time()
        model_predictions = self._recognize_parallel(cv_image, regions)
        rec_time = time.time() - rec_start
        
        logger.debug(f"Recognition completed in {rec_time:.3f}s")
        
        # Step 3: Language Identification
        lang_start = time.time()
        # Combine all recognized text for language detection
        all_text = " ".join([
            r.text for pred in model_predictions 
            for r in pred.regions if r.text
        ])
        detected_lang, lang_conf = self.language_id.identify(all_text)
        lang_time = time.time() - lang_start
        
        logger.debug(f"Detected language: {detected_lang.value} (conf: {lang_conf:.2f})")
        
        # Step 4: Ensemble Voting
        ensemble_start = time.time()
        ensemble_result = self.ensemble_voter.vote(
            model_predictions,
            cv_image.shape,
            detected_lang.value
        )
        ensemble_time = time.time() - ensemble_start
        
        logger.debug(f"Ensemble voting completed in {ensemble_time:.3f}s")
        
        # Step 5: Post-processing
        post_start = time.time()
        final_regions = self._post_process_regions(
            ensemble_result.final_regions, 
            detected_lang
        )
        post_time = time.time() - post_start
        
        # Step 6: Build full text
        full_text = " ".join([r.text for r in final_regions if r.text])
        
        # Step 7: Land record parsing
        structured_data = None
        if self.config.postprocess.get("enable_land_record_parsing", True):
            structured_data = self.land_parser.parse(full_text, detected_lang)
        
        # Build result
        result = OCRResult(
            image_path=str(image) if isinstance(image, str) else "array",
            regions=final_regions,
            full_text=full_text,
            detected_language=detected_lang,
            language_confidence=lang_conf,
            processing_time=time.time() - start_time,
            model_used="ensemble",
            structured_data=structured_data,
            metadata={
                "detection_time": det_time,
                "recognition_time": rec_time,
                "language_id_time": lang_time,
                "ensemble_time": ensemble_time,
                "postprocess_time": post_time,
                "num_regions": len(final_regions),
                "models_used": list(self.recognizers.keys())
            }
        )
        
        # Optional: Generate visualization
        if return_visualization or self.config.output.get("save_visualization"):
            viz = create_visualization(
                original_image,
                [r.to_dict() for r in final_regions],
                color_by="language",
                show_confidence=True,
                show_language=True
            )
            result.metadata["visualization"] = viz
            
            if self.config.output.get("save_visualization"):
                viz_dir = self.config.output.get("viz_dir", "outputs/visualizations")
                Path(viz_dir).mkdir(parents=True, exist_ok=True)
                save_visualization(viz, f"{viz_dir}/result_{int(time.time())}.jpg")
        
        # Optional: Extract crops
        if return_crops or self.config.output.get("save_crops"):
            crops = extract_crops(original_image, [r.to_dict() for r in final_regions])
            result.metadata["crops"] = crops
        
        logger.info(f"OCR completed in {result.processing_time:.3f}s | "
                   f"Lang: {detected_lang.value} | Regions: {len(final_regions)}")
        
        return result
    
    def _recognize_parallel(
        self, 
        image: np.ndarray, 
        regions: List[TextRegion]
    ) -> List[ModelPrediction]:
        """Run recognition in parallel across models"""
        predictions = []
        
        def recognize_with_model(model_name: str, recognizer: BaseRecognizer) -> ModelPrediction:
            start = time.time()
            try:
                recognized_regions = recognizer.recognize(image, regions)
                return ModelPrediction(
                    model_name=model_name,
                    regions=recognized_regions,
                    processing_time=time.time() - start
                )
            except Exception as e:
                logger.error(f"Recognizer {model_name} failed: {e}")
                return ModelPrediction(
                    model_name=model_name,
                    regions=regions,  # Return original regions with empty text
                    processing_time=time.time() - start,
                    metadata={"error": str(e)}
                )
        
        # Use ThreadPoolExecutor for I/O bound operations (model inference)
        with ThreadPoolExecutor(max_workers=len(self.recognizers)) as executor:
            futures = {
                executor.submit(recognize_with_model, name, rec): name 
                for name, rec in self.recognizers.items()
            }
            
            for future in as_completed(futures):
                predictions.append(future.result())
        
        return predictions
    
    def _post_process_regions(
        self, 
        regions: List[TextRegion], 
        language: LanguageCode
    ) -> List[TextRegion]:
        """Apply post-processing to regions"""
        processed = []
        
        for region in regions:
            # Clean text
            cleaned_text = clean_ocr_text(region.text, language)
            
            # Apply Indic correction
            if self.corrector and self.config.postprocess.get("enable_indic_correction", True):
                corrected_text = self.corrector.correct(cleaned_text, language)
            else:
                corrected_text = cleaned_text
            
            # Create new region with corrected text
            new_region = TextRegion(
                text=corrected_text,
                confidence=region.confidence,
                bbox=region.bbox,
                polygon=region.polygon,
                language=language,
                script=region.script,
                model_source=region.model_source,
                metadata=region.metadata
            )
            processed.append(new_region)
        
        return processed
    
    def process_batch(
        self,
        images: List[Union[str, np.ndarray, bytes]],
        max_workers: int = 4,
        **kwargs
    ) -> List[OCRResult]:
        """
        Process multiple images in parallel
        
        Args:
            images: List of images
            max_workers: Maximum parallel workers
            **kwargs: Additional arguments for process()
            
        Returns:
            List of OCRResult
        """
        results = []
        
        def process_single(img):
            return self.process(img, **kwargs)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single, img): img for img in images}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch processing failed for image: {e}")
                    results.append(None)
        
        return results
    
    def process_directory(
        self,
        input_dir: str,
        output_dir: str,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tiff", ".bmp"),
        **kwargs
    ) -> List[OCRResult]:
        """Process all images in a directory"""
        from pathlib import Path
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} images in {input_dir}")
        
        results = self.process_batch([str(f) for f in image_files], **kwargs)
        
        # Save results
        for result, img_file in zip(results, image_files):
            if result:
                out_file = output_path / f"{img_file.stem}_ocr.json"
                result.save_json(str(out_file))
        
        return results
    
    def benchmark(
        self,
        test_images: List[Tuple[str, str]],  # (image_path, ground_truth_text)
        languages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run benchmark on test set
        
        Args:
            test_images: List of (image_path, ground_truth_text)
            languages: Optional list of language codes
            
        Returns:
            Benchmark results
        """
        from .utils.metrics import compute_metrics, compute_metrics_by_language, print_metrics_table
        
        references = []
        hypotheses = []
        langs = []
        
        for img_path, gt_text in test_images:
            result = self.process(img_path)
            references.append(gt_text)
            hypotheses.append(result.full_text)
            langs.append(languages.pop(0) if languages else result.detected_language.value)
        
        if languages:
            metrics_by_lang = compute_metrics_by_language(references, hypotheses, langs)
            print_metrics_table(metrics_by_lang)
            return {lang: m.to_dict() for lang, m in metrics_by_lang.items()}
        else:
            overall = compute_metrics(references, hypotheses, langs)
            print(overall.summary())
            return overall.to_dict()


# Convenience functions
def create_ocr(
    config_path: Optional[str] = None,
    languages: Optional[List[str]] = None,
    device: str = "auto",
    **kwargs
) -> IndicOCR:
    """Factory function to create IndicOCR instance"""
    config = PipelineConfig()
    if languages:
        config.languages = languages
    config.device = device
    
    ocr = IndicOCR(config=config, **kwargs)
    ocr.initialize()
    return ocr


def quick_ocr(
    image: Union[str, np.ndarray, bytes],
    languages: Optional[List[str]] = None
) -> str:
    """Quick OCR - single function call"""
    ocr = create_ocr(languages=languages)
    result = ocr.process(image)
    return result.full_text