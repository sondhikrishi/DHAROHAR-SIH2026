#!/usr/bin/env python3
"""
IndicOCR Demo Script
====================
Command-line demo for the IndicOCR system.
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indic_ocr import IndicOCR, create_ocr, quick_ocr
from src.utils.visualization import save_visualization


def main():
    parser = argparse.ArgumentParser(
        description="IndicOCR - Multilingual Indian Language OCR Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python demo.py --image land_record.jpg
  
  # With visualization
  python demo.py --image land_record.jpg --visualize --save-crops
  
  # Batch process directory
  python demo.py --input-dir data/land_records/ --output results/
  
  # Specify languages
  python demo.py --image doc.jpg --languages hi en bn ta
  
  # Use config file
  python demo.py --image doc.jpg --config configs/model_config.yaml
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", "-i", help="Path to input image")
    input_group.add_argument("--input-dir", "-d", help="Input directory for batch processing")
    
    # Output options
    parser.add_argument("--output", "-o", default="outputs", help="Output directory")
    parser.add_argument("--format", choices=["json", "txt", "csv"], default="json", help="Output format")
    
    # Processing options
    parser.add_argument("--languages", "-l", nargs="+", 
                       default=["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"],
                       help="Language codes to enable")
    parser.add_argument("--config", "-c", help="Path to config YAML file")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto", help="Device to use")
    
    # Visualization options
    parser.add_argument("--visualize", "-v", action="store_true", help="Generate visualization")
    parser.add_argument("--save-crops", action="store_true", help="Save cropped text regions")
    parser.add_argument("--no-save-json", action="store_true", help="Don't save JSON output")
    
    # Other
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark on test set")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize OCR
    if not args.quiet:
        print("🔧 Initializing IndicOCR...")
    
    try:
        if args.config:
            ocr = IndicOCR(config_path=args.config, device=args.device)
        else:
            ocr = create_ocr(languages=args.languages, device=args.device)
        
        if not args.quiet:
            print("✅ IndicOCR ready!")
            print(f"   Models: {list(ocr.recognizers.keys())}")
            print(f"   Languages: {args.languages}")
            print(f"   Device: {ocr.device}")
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        sys.exit(1)
    
    # Process single image
    if args.image:
        process_single_image(ocr, args, output_dir)
    
    # Batch process directory
    elif args.input_dir:
        process_batch(ocr, args, output_dir)
    
    # Benchmark
    if args.benchmark:
        run_benchmark(ocr, args)


def process_single_image(ocr: IndicOCR, args, output_dir: Path):
    """Process a single image"""
    if not args.quiet:
        print(f"\n📸 Processing: {args.image}")
    
    start = time.time()
    result = ocr.process(
        args.image,
        return_visualization=args.visualize,
        return_crops=args.save_crops
    )
    elapsed = time.time() - start
    
    # Print results
    print(f"\n📝 Results:")
    print(f"   Language: {result.detected_language.value} (confidence: {result.language_confidence:.2f})")
    print(f"   Regions:  {len(result.regions)}")
    print(f"   Time:     {elapsed:.3f}s")
    print(f"   Text:     {result.full_text[:200]}{'...' if len(result.full_text) > 200 else ''}")
    
    if result.structured_data:
        print(f"\n🏠 Extracted Fields:")
        for field, data in result.structured_data.items():
            if isinstance(data, dict) and "value" in data:
                print(f"   {field}: {data['value']} (conf: {data.get('confidence', 0):.2f})")
    
    # Save outputs
    if not args.no_save_json:
        output_file = output_dir / f"{Path(args.image).stem}_result.json"
        result.save_json(str(output_file))
        if not args.quiet:
            print(f"\n💾 Saved JSON: {output_file}")
    
    # Save visualization
    if args.visualize and "visualization" in result.metadata:
        viz_file = output_dir / f"{Path(args.image).stem}_viz.jpg"
        save_visualization(result.metadata["visualization"], str(viz_file))
        if not args.quiet:
            print(f"💾 Saved visualization: {viz_file}")
    
    # Save crops
    if args.save_crops and "crops" in result.metadata:
        crops_dir = output_dir / "crops" / Path(args.image).stem
        crops_dir.mkdir(parents=True, exist_ok=True)
        for i, crop in enumerate(result.metadata["crops"]):
            crop_file = crops_dir / f"crop_{i:03d}.jpg"
            cv2.imwrite(str(crop_file), crop)
        if not args.quiet:
            print(f"💾 Saved {len(result.metadata['crops'])} crops to: {crops_dir}")
    
    # Save text only
    if args.format == "txt":
        txt_file = output_dir / f"{Path(args.image).stem}.txt"
        txt_file.write_text(result.full_text, encoding="utf-8")
        if not args.quiet:
            print(f"💾 Saved text: {txt_file}")


def process_batch(ocr: IndicOCR, args, output_dir: Path):
    """Batch process directory"""
    from pathlib import Path
    
    input_dir = Path(args.input_dir)
    extensions = (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp")
    
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ No images found in {input_dir}")
        return
    
    print(f"\n📁 Found {len(image_files)} images")
    print(f"🔄 Processing batch...")
    
    results = ocr.process_batch(
        [str(f) for f in image_files],
        max_workers=4,
        return_visualization=args.visualize,
        return_crops=args.save_crops
    )
    
    # Save results
    successful = 0
    for result, img_file in zip(results, image_files):
        if result is None:
            print(f"❌ Failed: {img_file.name}")
            continue
        
        successful += 1
        
        if not args.no_save_json:
            out_file = output_dir / f"{img_file.stem}_result.json"
            result.save_json(str(out_file))
        
        if args.visualize and "visualization" in result.metadata:
            viz_file = output_dir / f"{img_file.stem}_viz.jpg"
            save_visualization(result.metadata["visualization"], str(viz_file))
        
        if not args.quiet:
            print(f"✅ {img_file.name}: {result.detected_language.value} | "
                  f"{len(result.regions)} regions | {result.processing_time:.2f}s")
    
    print(f"\n📊 Batch complete: {successful}/{len(image_files)} successful")
    
    # Summary
    if successful > 0:
        avg_time = sum(r.processing_time for r in results if r) / successful
        lang_dist = {}
        for r in results:
            if r:
                lang_dist[r.detected_language.value] = lang_dist.get(r.detected_language.value, 0) + 1
        
        print(f"   Average time: {avg_time:.2f}s/image")
        print(f"   Language distribution: {lang_dist}")


def run_benchmark(ocr: IndicOCR, args):
    """Run benchmark (placeholder - needs test data)"""
    print("\n🏃 Benchmark mode - requires test data with ground truth")
    print("   Create test data in data/test/ with corresponding .txt files")
    
    test_dir = Path("data/test")
    if not test_dir.exists():
        print("   ❌ Test directory not found")
        return
    
    # Find test images with ground truth
    test_cases = []
    for img_file in test_dir.glob("*.jpg"):
        txt_file = img_file.with_suffix(".txt")
        if txt_file.exists():
            gt_text = txt_file.read_text(encoding="utf-8").strip()
            test_cases.append((str(img_file), gt_text))
    
    if not test_cases:
        print("   ❌ No test cases found (need .jpg + .txt pairs)")
        return
    
    print(f"   Found {len(test_cases)} test cases")
    results = ocr.benchmark(test_cases)
    print("   Benchmark complete!")


if __name__ == "__main__":
    # Import cv2 here to avoid issues if not installed
    try:
        import cv2
    except ImportError:
        print("❌ OpenCV not installed. Run: pip install opencv-python")
        sys.exit(1)
    
    main()