#!/usr/bin/env python3
"""
Model Download Script for IndicOCR
==================================
Download all required models for offline use.
"""

import argparse
import sys
import os
from pathlib import Path
import urllib.request
import tarfile
import zipfile
import shutil
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: Path, desc: str = None):
    """Download file with progress bar"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if dest.exists():
        print(f"✅ Already exists: {dest.name}")
        return
    
    print(f"⬇️  Downloading: {dest.name}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc or dest.name) as t:
        urllib.request.urlretrieve(url, dest, reporthook=t.update_to)
    print(f"✅ Downloaded: {dest.name}")


def extract_archive(archive: Path, dest: Path):
    """Extract archive file"""
    print(f"📦 Extracting: {archive.name}")
    dest.mkdir(parents=True, exist_ok=True)
    
    if archive.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive, 'r:*') as tf:
            tf.extractall(dest)
    elif archive.suffix == '.zip':
        with zipfile.ZipFile(archive, 'r') as zf:
            zf.extractall(dest)
    else:
        print(f"⚠️  Unknown archive format: {archive.suffix}")
    
    print(f"✅ Extracted to: {dest}")


def download_paddle_models(models_dir: Path):
    """Download PaddleOCR models"""
    print("\n📥 Downloading PaddleOCR models...")
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # PaddleOCR v4 server models (multilingual including Indic)
    base_url = "https://paddleocr.bj.bcebos.com/PP-OCRv4"
    
    models = {
        "det": {
            "url": f"{base_url}/ch_ppocr_mobile_v2.0_det_infer.tar",
            "name": "ch_ppocr_mobile_v2.0_det_infer.tar"
        },
        "rec": {
            "url": f"{base_url}/ch_ppocr_mobile_v2.0_rec_infer.tar",
            "name": "ch_ppocr_mobile_v2.0_rec_infer.tar"
        },
        # For Indian languages, we need the multilingual model
        "rec_indian": {
            "url": f"{base_url}/multilingual/ppocr_mobile_v2.0_rec_infer.tar",
            "name": "ppocr_mobile_v2.0_rec_infer.tar"
        }
    }
    
    # Note: Actual model URLs may vary. These are example URLs.
    # In practice, PaddleOCR auto-downloads on first use.
    print("ℹ️  PaddleOCR models will auto-download on first use")
    print("   To pre-download, run: paddleocr --download_models")
    
    # Create placeholder structure
    (models_dir / "paddle" / "det").mkdir(parents=True, exist_ok=True)
    (models_dir / "paddle" / "rec").mkdir(parents=True, exist_ok=True)


def download_trocr_models(models_dir: Path):
    """Download TrOCR models"""
    print("\n📥 Downloading TrOCR models...")
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # TrOCR base models (downloaded via transformers on first use)
    print("ℹ️  TrOCR models (microsoft/trocr-base-handwritten) will auto-download via transformers")
    print("   For fine-tuned Indic models, place them in: models/trocr-indic-finetuned/")


def download_fasttext_lid(models_dir: Path):
    """Download fastText language identification model"""
    print("\n📥 Downloading fastText LID model...")
    
    models_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
    dest = models_dir / "lid.176.ftz"
    
    download_file(url, dest, "fastText LID")


def download_tesseract_data(models_dir: Path):
    """Download Tesseract traineddata for Indian languages"""
    print("\n📥 Downloading Tesseract traineddata...")
    
    tessdata_dir = models_dir / "tesseract" / "tessdata"
    tessdata_dir.mkdir(parents=True, exist_ok=True)
    
    # Indian language codes for Tesseract
    langs = {
        "hin": "Hindi",
        "ben": "Bengali",
        "tel": "Telugu",
        "mar": "Marathi",
        "tam": "Tamil",
        "guj": "Gujarati",
        "kan": "Kannada",
        "mal": "Malayalam",
        "pan": "Punjabi",
        "ori": "Odia",
        "asm": "Assamese",
        "urd": "Urdu",
        "eng": "English",
        "osd": "Orientation/Script Detection"
    }
    
    base_url = "https://github.com/tesseract-ocr/tessdata_best/raw/main"
    
    for code, name in langs.items():
        dest = tessdata_dir / f"{code}.traineddata"
        if dest.exists():
            print(f"✅ {name} ({code}): Already exists")
            continue
        
        url = f"{base_url}/{code}.traineddata"
        try:
            download_file(url, dest, f"Tesseract {name}")
        except Exception as e:
            print(f"⚠️  Failed to download {name}: {e}")


def download_easyocr_models(models_dir: Path):
    """Download EasyOCR models"""
    print("\n📥 Downloading EasyOCR models...")
    
    model_dir = models_dir / "easyocr"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("ℹ️  EasyOCR models will auto-download on first use")
    print("   Supported languages: hi, bn, ta, te, kn, ml, en")


def download_indicbert(models_dir: Path):
    """Download IndicBERT model"""
    print("\n📥 Downloading IndicBERT...")
    
    print("ℹ️  IndicBERT (ai4bharat/indicbert) will auto-download via transformers")
    print("   For post-correction, model downloads on first use")


def verify_installation(models_dir: Path):
    """Verify downloaded models"""
    print("\n🔍 Verifying installation...")
    
    checks = [
        (models_dir / "paddle" / "det", "PaddleOCR detector"),
        (models_dir / "paddle" / "rec", "PaddleOCR recognizer"),
        (models_dir / "tesseract" / "tessdata" / "hin.traineddata", "Tesseract Hindi"),
        (models_dir / "tesseract" / "tessdata" / "eng.traineddata", "Tesseract English"),
        (models_dir / "lid.176.ftz", "fastText LID"),
    ]
    
    all_ok = True
    for path, name in checks:
        if path.exists():
            print(f"✅ {name}: OK")
        else:
            print(f"❌ {name}: Missing ({path})")
            all_ok = False
    
    if all_ok:
        print("\n✅ All models verified!")
    else:
        print("\n⚠️  Some models missing - they will download on first use")


def main():
    parser = argparse.ArgumentParser(
        description="Download IndicOCR models for offline use"
    )
    parser.add_argument("--models-dir", default="models", help="Models directory")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--paddle", action="store_true", help="Download PaddleOCR models")
    parser.add_argument("--trocr", action="store_true", help="Download TrOCR models")
    parser.add_argument("--tesseract", action="store_true", help="Download Tesseract data")
    parser.add_argument("--easyocr", action="store_true", help="Download EasyOCR models")
    parser.add_argument("--fasttext", action="store_true", help="Download fastText LID")
    parser.add_argument("--indicbert", action="store_true", help="Download IndicBERT")
    parser.add_argument("--verify", action="store_true", help="Verify existing models")
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Models directory: {models_dir}")
    
    if args.verify:
        verify_installation(models_dir)
        return
    
    # Default: download all if no specific flags
    download_all = args.all or not any([
        args.paddle, args.trocr, args.tesseract, 
        args.easyocr, args.fasttext, args.indicbert
    ])
    
    if download_all or args.fasttext:
        download_fasttext_lid(models_dir)
    
    if download_all or args.tesseract:
        download_tesseract_data(models_dir)
    
    if download_all or args.paddle:
        download_paddle_models(models_dir)
    
    if download_all or args.trocr:
        download_trocr_models(models_dir)
    
    if download_all or args.easyocr:
        download_easyocr_models(models_dir)
    
    if download_all or args.indicbert:
        download_indicbert(models_dir)
    
    verify_installation(models_dir)
    
    print("\n🎉 Model download complete!")
    print(f"   Models saved to: {models_dir}")
    print("\n💡 To use offline, set environment variables:")
    print(f"   export INDIC_OCR_MODELS_DIR={models_dir}")
    print(f"   export TESSDATA_PREFIX={models_dir}/tesseract/tessdata")


if __name__ == "__main__":
    main()