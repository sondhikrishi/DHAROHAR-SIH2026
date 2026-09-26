#!/usr/bin/env python3
"""
FastAPI Server for IndicOCR
===========================
REST API for OCR inference.
"""

import io
import base64
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Union

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from indic_ocr import IndicOCR, create_ocr
from src.data_models import OCRResult, LanguageCode

app = FastAPI(
    title="IndicOCR API",
    description="Multilingual Indian Language OCR for Land Records",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global OCR instance
ocr_instance: Optional[IndicOCR] = None


class OCRRequest(BaseModel):
    """OCR request model"""
    languages: Optional[List[str]] = Field(
        default=["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"],
        description="Language codes to enable"
    )
    return_visualization: bool = Field(default=False, description="Return visualization image")
    return_crops: bool = Field(default=False, description="Return cropped regions")
    return_structured: bool = Field(default=True, description="Extract land record fields")


class OCRResponse(BaseModel):
    """OCR response model"""
    success: bool
    text: str
    language: str
    language_confidence: float
    regions: int
    processing_time: float
    structured_data: Optional[dict] = None
    visualization: Optional[str] = None  # Base64 encoded
    crops: Optional[List[str]] = None  # Base64 encoded
    error: Optional[str] = None


class BatchOCRResponse(BaseModel):
    """Batch OCR response"""
    success: bool
    results: List[OCRResponse]
    total_time: float
    processed: int
    failed: int


@app.on_event("startup")
async def startup_event():
    """Initialize OCR on startup"""
    global ocr_instance
    print("🔧 Initializing IndicOCR...")
    try:
        ocr_instance = create_ocr()
        print("✅ IndicOCR ready!")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        ocr_instance = None


@app.get("/")
async def root():
    return {
        "name": "IndicOCR API",
        "version": "1.0.0",
        "description": "Multilingual Indian Language OCR for Land Records",
        "docs": "/docs",
        "models": list(ocr_instance.recognizers.keys()) if ocr_instance else [],
        "languages": ocr_instance.config.languages if ocr_instance else []
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if ocr_instance else "unhealthy",
        "models_loaded": len(ocr_instance.recognizers) if ocr_instance else 0,
        "device": ocr_instance.device if ocr_instance else "unknown"
    }


@app.post("/ocr", response_model=OCRResponse)
async def ocr_single(
    file: UploadFile = File(...),
    languages: Optional[str] = Form(None),
    return_visualization: bool = Form(False),
    return_crops: bool = Form(False),
    return_structured: bool = Form(True)
):
    """
    Process a single image
    
    - **file**: Image file (JPG, PNG, TIFF, BMP)
    - **languages**: Comma-separated language codes (e.g., "hi,en,bn")
    - **return_visualization**: Return annotated image as base64
    - **return_crops**: Return cropped text regions as base64
    - **return_structured**: Extract land record fields
    """
    if not ocr_instance:
        raise HTTPException(status_code=503, detail="OCR not initialized")
    
    # Validate file
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Parse languages
    lang_list = None
    if languages:
        lang_list = [l.strip() for l in languages.split(",")]
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Process
        start = time.time()
        result = ocr_instance.process(
            tmp_path,
            return_visualization=return_visualization,
            return_crops=return_crops
        )
        
        # Prepare response
        response = OCRResponse(
            success=True,
            text=result.full_text,
            language=result.detected_language.value,
            language_confidence=result.language_confidence,
            regions=len(result.regions),
            processing_time=time.time() - start,
            structured_data=result.structured_data if return_structured else None
        )
        
        # Add visualization if requested
        if return_visualization and "visualization" in result.metadata:
            import cv2
            _, buffer = cv2.imencode(".jpg", result.metadata["visualization"])
            response.visualization = base64.b64encode(buffer).decode("utf-8")
        
        # Add crops if requested
        if return_crops and "crops" in result.metadata:
            import cv2
            crops_b64 = []
            for crop in result.metadata["crops"]:
                _, buffer = cv2.imencode(".jpg", crop)
                crops_b64.append(base64.b64encode(buffer).decode("utf-8"))
            response.crops = crops_b64
        
        return response
        
    except Exception as e:
        return OCRResponse(
            success=False,
            text="",
            language="",
            language_confidence=0.0,
            regions=0,
            processing_time=0.0,
            error=str(e)
        )
    finally:
        # Cleanup
        try:
            Path(tmp_path).unlink()
        except:
            pass


@app.post("/ocr/batch", response_model=BatchOCRResponse)
async def ocr_batch(
    files: List[UploadFile] = File(...),
    languages: Optional[str] = Form(None),
    return_visualization: bool = Form(False),
    return_crops: bool = Form(False),
    return_structured: bool = Form(True),
    max_workers: int = Form(4)
):
    """
    Process multiple images in parallel
    """
    if not ocr_instance:
        raise HTTPException(status_code=503, detail="OCR not initialized")
    
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 files per batch")
    
    # Parse languages
    lang_list = None
    if languages:
        lang_list = [l.strip() for l in languages.split(",")]
    
    # Save all files temporarily
    tmp_paths = []
    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_paths.append(tmp.name)
    
    try:
        start = time.time()
        results = ocr_instance.process_batch(
            tmp_paths,
            max_workers=max_workers,
            return_visualization=return_visualization,
            return_crops=return_crops
        )
        
        # Build responses
        responses = []
        for i, result in enumerate(results):
            if result is None:
                responses.append(OCRResponse(
                    success=False,
                    text="",
                    language="",
                    language_confidence=0.0,
                    regions=0,
                    processing_time=0.0,
                    error="Processing failed"
                ))
                continue
            
            resp = OCRResponse(
                success=True,
                text=result.full_text,
                language=result.detected_language.value,
                language_confidence=result.language_confidence,
                regions=len(result.regions),
                processing_time=result.processing_time,
                structured_data=result.structured_data if return_structured else None
            )
            
            if return_visualization and "visualization" in result.metadata:
                import cv2
                _, buffer = cv2.imencode(".jpg", result.metadata["visualization"])
                resp.visualization = base64.b64encode(buffer).decode("utf-8")
            
            if return_crops and "crops" in result.metadata:
                import cv2
                crops_b64 = []
                for crop in result.metadata["crops"]:
                    _, buffer = cv2.imencode(".jpg", crop)
                    crops_b64.append(base64.b64encode(buffer).decode("utf-8"))
                resp.crops = crops_b64
            
            responses.append(resp)
        
        return BatchOCRResponse(
            success=True,
            results=responses,
            total_time=time.time() - start,
            processed=sum(1 for r in responses if r.success),
            failed=sum(1 for r in responses if not r.success)
        )
        
    finally:
        # Cleanup
        for tmp_path in tmp_paths:
            try:
                Path(tmp_path).unlink()
            except:
                pass


@app.post("/ocr/base64", response_model=OCRResponse)
async def ocr_base64(
    image_base64: str = Form(...),
    filename: str = Form("image.jpg"),
    languages: Optional[str] = Form(None),
    return_visualization: bool = Form(False),
    return_crops: bool = Form(False),
    return_structured: bool = Form(True)
):
    """
    Process image from base64 string
    """
    if not ocr_instance:
        raise HTTPException(status_code=503, detail="OCR not initialized")
    
    # Decode base64
    try:
        image_data = base64.b64decode(image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")
    
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
        tmp.write(image_data)
        tmp_path = tmp.name
    
    try:
        lang_list = None
        if languages:
            lang_list = [l.strip() for l in languages.split(",")]
        
        start = time.time()
        result = ocr_instance.process(
            tmp_path,
            return_visualization=return_visualization,
            return_crops=return_crops
        )
        
        response = OCRResponse(
            success=True,
            text=result.full_text,
            language=result.detected_language.value,
            language_confidence=result.language_confidence,
            regions=len(result.regions),
            processing_time=time.time() - start,
            structured_data=result.structured_data if return_structured else None
        )
        
        if return_visualization and "visualization" in result.metadata:
            import cv2
            _, buffer = cv2.imencode(".jpg", result.metadata["visualization"])
            response.visualization = base64.b64encode(buffer).decode("utf-8")
        
        if return_crops and "crops" in result.metadata:
            import cv2
            crops_b64 = []
            for crop in result.metadata["crops"]:
                _, buffer = cv2.imencode(".jpg", crop)
                crops_b64.append(base64.b64encode(buffer).decode("utf-8"))
            response.crops = crops_b64
        
        return response
        
    except Exception as e:
        return OCRResponse(
            success=False,
            text="",
            language="",
            language_confidence=0.0,
            regions=0,
            processing_time=0.0,
            error=str(e)
        )
    finally:
        try:
            Path(tmp_path).unlink()
        except:
            pass


@app.get("/languages")
async def get_languages():
    """Get supported languages"""
    if not ocr_instance:
        raise HTTPException(status_code=503, detail="OCR not initialized")
    
    # Load language mapping
    import yaml
    with open("configs/language_mapping.yaml", "r") as f:
        lang_config = yaml.safe_load(f)
    
    return {
        "supported": ocr_instance.config.languages,
        "details": lang_config.get("languages", {})
    }


def main():
    parser = argparse.ArgumentParser(description="IndicOCR API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--workers", type=int, default=1, help="Worker processes")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    
    args = parser.parse_args()
    
    print(f"🚀 Starting IndicOCR API on {args.host}:{args.port}")
    print(f"   Docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload
    )


if __name__ == "__main__":
    main()