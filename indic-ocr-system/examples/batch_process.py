#!/usr/bin/env python3
"""
Batch Processing Script for IndicOCR
====================================
Process large numbers of images efficiently.
"""

import argparse
import sys
import os
import json
import csv
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
from dataclasses import asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from indic_ocr import IndicOCR, create_ocr
from src.data_models import OCRResult


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch process images with IndicOCR",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input_dir", help="Input directory with images")
    parser.add_argument("output_dir", help="Output directory for results")
    
    parser.add_argument("--extensions", nargs="+", 
                       default=[".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"],
                       help="Image file extensions to process")
    parser.add_argument("--languages", "-l", nargs="+",
                       default=["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"])
    parser.add_argument("--config", "-c", help="Config YAML file")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    
    parser.add_argument("--workers", "-w", type=int, default=4, help="Parallel workers")
    parser.add_argument("--batch-size", type=int, default=10, help="Images per batch")
    
    parser.add_argument("--output-format", choices=["json", "jsonl", "csv", "txt"], default="json")
    parser.add_argument("--save-visualization", action="store_true", help="Save visualization images")
    parser.add_argument("--save-crops", action="store_true", help="Save cropped text regions")
    
    parser.add_argument("--recursive", "-r", action="store_true", help="Process subdirectories")
    parser.add_argument("--resume", action="store_true", help="Skip already processed images")
    parser.add_argument("--limit", type=int, help="Maximum images to process")
    
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    parser.add_argument("--log-file", help="Log file path")
    
    return parser.parse_args()


def find_images(input_dir: Path, extensions: List[str], recursive: bool) -> List[Path]:
    """Find all image files in directory"""
    images = []
    pattern = "**/*" if recursive else "*"
    
    for ext in extensions:
        images.extend(input_dir.glob(f"{pattern}{ext}"))
        images.extend(input_dir.glob(f"{pattern}{ext.upper()}"))
    
    return sorted(images)


def save_result(result: OCRResult, output_dir: Path, format: str, 
                save_viz: bool, save_crops: bool, base_name: str):
    """Save OCR result in specified format"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        out_file = output_dir / f"{base_name}.json"
        result.save_json(str(out_file))
    
    elif format == "jsonl":
        out_file = output_dir / "results.jsonl"
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    
    elif format == "csv":
        out_file = output_dir / "results.csv"
        # Flatten for CSV
        row = {
            "image": base_name,
            "language": result.detected_language.value,
            "language_confidence": result.language_confidence,
            "num_regions": len(result.regions),
            "processing_time": result.processing_time,
            "full_text": result.full_text,
        }
        if result.structured_data:
            for field, data in result.structured_data.items():
                if isinstance(data, dict):
                    row[f"field_{field}"] = data.get("value", "")
                    row[f"field_{field}_conf"] = data.get("confidence", 0)
        
        file_exists = out_file.exists()
        with open(out_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    
    elif format == "txt":
        out_file = output_dir / f"{base_name}.txt"
        out_file.write_text(result.full_text, encoding="utf-8")
    
    # Save visualization
    if save_viz and "visualization" in result.metadata:
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        from src.utils.visualization import save_visualization
        save_visualization(result.metadata["visualization"], str(viz_dir / f"{base_name}_viz.jpg"))
    
    # Save crops
    if save_crops and "crops" in result.metadata:
        crops_dir = output_dir / "crops" / base_name
        crops_dir.mkdir(parents=True, exist_ok=True)
        import cv2
        for i, crop in enumerate(result.metadata["crops"]):
            cv2.imwrite(str(crops_dir / f"crop_{i:03d}.jpg"), crop)


def process_batch_worker(ocr: IndicOCR, image_paths: List[Path], args) -> List[Dict]:
    """Worker function for batch processing"""
    results = []
    
    for img_path in image_paths:
        try:
            result = ocr.process(
                str(img_path),
                return_visualization=args.save_visualization,
                return_crops=args.save_crops
            )
            results.append({
                "path": img_path,
                "result": result,
                "error": None
            })
        except Exception as e:
            results.append({
                "path": img_path,
                "result": None,
                "error": str(e)
            })
    
    return results


def main():
    args = parse_args()
    
    # Setup logging
    if args.log_file:
        import logging
        logging.basicConfig(
            filename=args.log_file,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"❌ Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    images = find_images(input_dir, args.extensions, args.recursive)
    
    if args.limit:
        images = images[:args.limit]
    
    if not images:
        print("❌ No images found")
        sys.exit(1)
    
    # Filter already processed if resume
    if args.resume:
        processed = set()
        if args.output_format == "json":
            for f in output_dir.glob("*.json"):
                processed.add(f.stem.replace("_result", ""))
        elif args.output_format == "jsonl":
            # Would need to parse jsonl
            pass
        
        images = [img for img in images if img.stem not in processed]
        print(f"📋 Resuming: {len(images)} images remaining")
    
    print(f"📁 Found {len(images)} images to process")
    print(f"⚙️  Workers: {args.workers} | Batch size: {args.batch_size}")
    
    # Initialize OCR
    if not args.quiet:
        print("🔧 Initializing IndicOCR...")
    
    if args.config:
        ocr = IndicOCR(config_path=args.config, device=args.device)
    else:
        ocr = create_ocr(languages=args.languages, device=args.device)
    
    if not args.quiet:
        print("✅ Ready!")
    
    # Process in batches
    all_results = []
    start_time = time.time()
    
    # Split into chunks for workers
    chunks = [images[i:i + args.batch_size] for i in range(0, len(images), args.batch_size)]
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_batch_worker, ocr, chunk, args): chunk 
            for chunk in chunks
        }
        
        completed = 0
        for future in as_completed(futures):
            chunk_results = future.result()
            
            for item in chunk_results:
                img_path = item["path"]
                result = item["result"]
                error = item["error"]
                
                if error:
                    print(f"❌ {img_path.name}: {error}")
                    if args.log_file:
                        import logging
                        logging.error(f"{img_path}: {error}")
                    continue
                
                # Save result
                save_result(result, output_dir, args.output_format, 
                          args.save_visualization, args.save_crops, img_path.stem)
                
                all_results.append(result)
                completed += 1
                
                if not args.quiet:
                    print(f"✅ [{completed}/{len(images)}] {img_path.name}: "
                          f"{result.detected_language.value} | "
                          f"{len(result.regions)} regions | "
                          f"{result.processing_time:.2f}s")
    
    # Summary
    elapsed = time.time() - start_time
    successful = len([r for r in all_results if r is not None])
    
    print(f"\n📊 Batch Processing Complete")
    print(f"   Total:     {len(images)}")
    print(f"   Success:   {successful}")
    print(f"   Failed:    {len(images) - successful}")
    print(f"   Time:      {elapsed:.1f}s ({elapsed/len(images):.2f}s/image)" if images else "")
    
    if successful > 0:
        avg_time = sum(r.processing_time for r in all_results) / successful
        lang_dist = {}
        for r in all_results:
            lang_dist[r.detected_language.value] = lang_dist.get(r.detected_language.value, 0) + 1
        print(f"   Avg time:  {avg_time:.2f}s")
        print(f"   Languages: {lang_dist}")
    
    # Save summary
    summary = {
        "total_images": len(images),
        "successful": successful,
        "failed": len(images) - successful,
        "total_time": elapsed,
        "avg_time_per_image": elapsed / len(images) if images else 0,
        "language_distribution": lang_dist if successful > 0 else {},
        "config": {
            "languages": args.languages,
            "device": args.device,
            "workers": args.workers,
            "batch_size": args.batch_size
        }
    }
    
    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Summary saved: {summary_file}")


if __name__ == "__main__":
    main()