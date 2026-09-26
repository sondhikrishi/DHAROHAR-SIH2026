#!/usr/bin/env python3
"""
Benchmark Script for IndicOCR
=============================
Evaluate OCR accuracy on test datasets.
"""

import argparse
import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dataclasses import asdict

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from indic_ocr import IndicOCR, create_ocr
from src.utils.metrics import (
    compute_metrics, compute_metrics_by_language, print_metrics_table,
    calculate_field_accuracy, OCRMetrics
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark IndicOCR on test dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Data Format:
  data/test/
    ├── image1.jpg
    ├── image1.txt          # Ground truth text
    ├── image1.json         # Optional: structured ground truth
    ├── image2.jpg
    ├── image2.txt
    └── ...

JSON Ground Truth Format:
  {
    "full_text": "complete ground truth text",
    "language": "hi",
    "fields": {
      "survey_number": {"value": "142/2A", "confidence": 1.0},
      "owner_name": {"value": "राम कुमार", "confidence": 1.0}
    }
  }
        """
    )
    
    parser.add_argument("test_dir", help="Test directory with images and ground truth")
    parser.add_argument("--output", "-o", default="benchmark_results", help="Output directory")
    
    parser.add_argument("--config", "-c", help="Config YAML file")
    parser.add_argument("--languages", "-l", nargs="+",
                       default=["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"])
    parser.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    
    parser.add_argument("--models", nargs="+", 
                       default=["paddle_ocr", "trocr", "tesseract", "easyocr", "ensemble"],
                       help="Models to benchmark")
    parser.add_argument("--limit", type=int, help="Limit number of test images")
    
    parser.add_argument("--per-image", action="store_true", help="Save per-image metrics")
    parser.add_argument("--plots", action="store_true", help="Generate visualization plots")
    
    return parser.parse_args()


def load_test_data(test_dir: Path, limit: int = None) -> List[Tuple[str, str, str, Dict]]:
    """
    Load test data from directory
    
    Returns:
        List of (image_path, ground_truth_text, language, structured_gt)
    """
    test_cases = []
    
    # Find image files
    extensions = [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"]
    image_files = []
    for ext in extensions:
        image_files.extend(test_dir.glob(f"*{ext}"))
        image_files.extend(test_dir.glob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files)
    
    if limit:
        image_files = image_files[:limit]
    
    for img_file in image_files:
        # Load ground truth text
        txt_file = img_file.with_suffix(".txt")
        if not txt_file.exists():
            print(f"⚠️  No ground truth for {img_file.name}, skipping")
            continue
        
        gt_text = txt_file.read_text(encoding="utf-8").strip()
        
        # Load language (from filename or separate file)
        lang_file = img_file.with_suffix(".lang")
        if lang_file.exists():
            language = lang_file.read_text().strip()
        else:
            # Try to infer from parent directory name
            language = img_file.parent.name
            if language not in ["hi", "en", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as", "ur"]:
                language = "hi"  # Default
        
        # Load structured ground truth
        structured_gt = {}
        json_file = img_file.with_suffix(".json")
        if json_file.exists():
            try:
                structured_gt = json.loads(json_file.read_text(encoding="utf-8"))
            except:
                pass
        
        test_cases.append((str(img_file), gt_text, language, structured_gt))
    
    return test_cases


def run_model_benchmark(
    ocr: IndicOCR,
    test_cases: List[Tuple[str, str, str, Dict]],
    model_name: str
) -> OCRMetrics:
    """Run benchmark for a specific model"""
    references = []
    hypotheses = []
    languages = []
    per_image = []
    
    for img_path, gt_text, gt_lang, _ in test_cases:
        # For individual models, we'd need to modify the pipeline
        # For now, use ensemble and note which model contributed most
        result = ocr.process(img_path)
        
        references.append(gt_text)
        hypotheses.append(result.full_text)
        languages.append(gt_lang)
        
        per_image.append({
            "image": Path(img_path).name,
            "ground_truth": gt_text,
            "prediction": result.full_text,
            "language": gt_lang,
            "detected_language": result.detected_language.value,
            "processing_time": result.processing_time,
            "num_regions": len(result.regions)
        })
    
    metrics = compute_metrics(references, hypotheses, languages)
    metrics.per_sample = per_image
    
    return metrics


def run_ensemble_benchmark(
    ocr: IndicOCR,
    test_cases: List[Tuple[str, str, str, Dict]]
) -> OCRMetrics:
    """Run benchmark using ensemble (default)"""
    references = []
    hypotheses = []
    languages = []
    per_image = []
    
    print(f"\n🏃 Running ensemble benchmark on {len(test_cases)} images...")
    
    for i, (img_path, gt_text, gt_lang, _) in enumerate(test_cases):
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(test_cases)}")
        
        result = ocr.process(img_path)
        
        references.append(gt_text)
        hypotheses.append(result.full_text)
        languages.append(gt_lang)
        
        per_image.append({
            "image": Path(img_path).name,
            "ground_truth": gt_text,
            "prediction": result.full_text,
            "language": gt_lang,
            "detected_language": result.detected_language.value,
            "processing_time": result.processing_time,
            "num_regions": len(result.regions)
        })
    
    metrics = compute_metrics(references, hypotheses, languages)
    metrics.per_sample = per_image
    
    return metrics


def run_field_benchmark(
    ocr: IndicOCR,
    test_cases: List[Tuple[str, str, str, Dict]]
) -> Dict[str, float]:
    """Benchmark field extraction accuracy"""
    print("\n🏠 Running field extraction benchmark...")
    
    all_field_accuracies = {}
    field_counts = {}
    
    for img_path, _, gt_lang, structured_gt in test_cases:
        if not structured_gt or "fields" not in structured_gt:
            continue
        
        result = ocr.process(img_path)
        
        if not result.structured_data:
            continue
        
        # Extract predicted fields
        pred_fields = {}
        for field, data in result.structured_data.items():
            if isinstance(data, dict) and "value" in data:
                pred_fields[field] = data["value"]
        
        gt_fields = {}
        for field, data in structured_gt["fields"].items():
            if isinstance(data, dict) and "value" in data:
                gt_fields[field] = data["value"]
        
        # Calculate field accuracies
        field_accs = calculate_field_accuracy(gt_fields, pred_fields)
        
        for field, acc in field_accs.items():
            if field not in all_field_accuracies:
                all_field_accuracies[field] = []
                field_counts[field] = 0
            all_field_accuracies[field].append(acc)
            field_counts[field] += 1
    
    # Average per field
    avg_field_acc = {}
    for field, accs in all_field_accuracies.items():
        avg_field_acc[field] = {
            "accuracy": np.mean(accs),
            "std": np.std(accs),
            "count": field_counts[field]
        }
    
    return avg_field_acc


def save_results(
    output_dir: Path,
    ensemble_metrics: OCRMetrics,
    field_metrics: Dict[str, Any],
    per_image: bool
):
    """Save benchmark results"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Overall metrics
    with open(output_dir / "overall_metrics.json", "w") as f:
        json.dump(ensemble_metrics.to_dict(), f, indent=2, ensure_ascii=False)
    
    # Per-language metrics
    from src.utils.metrics import compute_metrics_by_language
    refs = [s["ground_truth"] for s in ensemble_metrics.per_sample]
    hyps = [s["prediction"] for s in ensemble_metrics.per_sample]
    langs = [s["language"] for s in ensemble_metrics.per_sample]
    
    lang_metrics = compute_metrics_by_language(refs, hyps, langs)
    lang_results = {lang: m.to_dict() for lang, m in lang_metrics.items()}
    
    with open(output_dir / "per_language_metrics.json", "w") as f:
        json.dump(lang_results, f, indent=2, ensure_ascii=False)
    
    # Field metrics
    with open(output_dir / "field_metrics.json", "w") as f:
        json.dump(field_metrics, f, indent=2, ensure_ascii=False)
    
    # Per-image details
    if per_image:
        with open(output_dir / "per_image_details.json", "w") as f:
            json.dump(ensemble_metrics.per_sample, f, indent=2, ensure_ascii=False)
    
    # CSV summary
    import csv
    with open(output_dir / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["CER", f"{ensemble_metrics.cer:.4f}"])
        writer.writerow(["WER", f"{ensemble_metrics.wer:.4f}"])
        writer.writerow(["F1", f"{ensemble_metrics.f1:.4f}"])
        writer.writerow(["Accuracy", f"{ensemble_metrics.accuracy:.4f}"])
        writer.writerow(["Num Samples", ensemble_metrics.num_samples])
        writer.writerow([])
        writer.writerow(["Language", "Samples", "CER", "WER", "F1", "Accuracy"])
        for lang, metrics in lang_metrics.items():
            writer.writerow([lang, metrics.num_samples, 
                           f"{metrics.cer:.4f}", f"{metrics.wer:.4f}", 
                           f"{metrics.f1:.4f}", f"{metrics.accuracy:.4f}"])
    
    print(f"\n💾 Results saved to: {output_dir}")


def generate_plots(output_dir: Path, ensemble_metrics: OCRMetrics, lang_metrics: Dict):
    """Generate visualization plots"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Per-language metrics bar chart
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        langs = list(lang_metrics.keys())
        if langs:
            cer_vals = [lang_metrics[l].cer for l in langs]
            wer_vals = [lang_metrics[l].wer for l in langs]
            f1_vals = [lang_metrics[l].f1 for l in langs]
            acc_vals = [lang_metrics[l].accuracy for l in langs]
            samples = [lang_metrics[l].num_samples for l in langs]
            
            x = range(len(langs))
            width = 0.2
            
            axes[0, 0].bar([i - 1.5*width for i in x], cer_vals, width, label='CER', color='#e74c3c')
            axes[0, 0].bar([i - 0.5*width for i in x], wer_vals, width, label='WER', color='#f39c12')
            axes[0, 0].bar([i + 0.5*width for i in x], f1_vals, width, label='F1', color='#2ecc71')
            axes[0, 0].bar([i + 1.5*width for i in x], acc_vals, width, label='Acc', color='#3498db')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(langs, rotation=45)
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_title('Per-Language Metrics')
            axes[0, 0].legend()
            axes[0, 0].set_ylim(0, 1.1)
            
            # Sample count
            axes[0, 1].bar(langs, samples, color='#9b59b6')
            axes[0, 1].set_ylabel('Number of Samples')
            axes[0, 1].set_title('Test Samples per Language')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Processing time distribution
            times = [s["processing_time"] for s in ensemble_metrics.per_sample]
            axes[1, 0].hist(times, bins=20, color='#34495e', alpha=0.7, edgecolor='white')
            axes[1, 0].axvline(np.mean(times), color='red', linestyle='--', label=f'Mean: {np.mean(times):.2f}s')
            axes[1, 0].set_xlabel('Processing Time (s)')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Processing Time Distribution')
            axes[1, 0].legend()
            
            # Region count distribution
            regions = [s["num_regions"] for s in ensemble_metrics.per_sample]
            axes[1, 1].hist(regions, bins=20, color='#16a085', alpha=0.7, edgecolor='white')
            axes[1, 1].axvline(np.mean(regions), color='red', linestyle='--', label=f'Mean: {np.mean(regions):.1f}')
            axes[1, 1].set_xlabel('Number of Text Regions')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Text Regions per Image')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "benchmark_plots.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plots saved to: {output_dir / 'benchmark_plots.png'}")
        
    except ImportError:
        print("⚠️  Matplotlib/Seaborn not installed, skipping plots")


def main():
    args = parse_args()
    
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output)
    
    if not test_dir.exists():
        print(f"❌ Test directory not found: {test_dir}")
        sys.exit(1)
    
    # Load test data
    print(f"📂 Loading test data from: {test_dir}")
    test_cases = load_test_data(test_dir, args.limit)
    
    if not test_cases:
        print("❌ No valid test cases found")
        sys.exit(1)
    
    print(f"✅ Loaded {len(test_cases)} test cases")
    
    # Language distribution
    lang_dist = {}
    for _, _, lang, _ in test_cases:
        lang_dist[lang] = lang_dist.get(lang, 0) + 1
    print(f"   Languages: {lang_dist}")
    
    # Initialize OCR
    print("🔧 Initializing IndicOCR...")
    if args.config:
        ocr = IndicOCR(config_path=args.config, device=args.device)
    else:
        ocr = create_ocr(languages=args.languages, device=args.device)
    
    # Run ensemble benchmark
    ensemble_metrics = run_ensemble_benchmark(ocr, test_cases)
    
    # Run field benchmark
    field_metrics = run_field_benchmark(ocr, test_cases)
    
    # Print results
    print("\n" + "="*60)
    print("📊 ENSEMBLE BENCHMARK RESULTS")
    print("="*60)
    print(ensemble_metrics.summary())
    
    print("\n📊 PER-LANGUAGE BREAKDOWN")
    print("-"*60)
    refs = [s["ground_truth"] for s in ensemble_metrics.per_sample]
    hyps = [s["prediction"] for s in ensemble_metrics.per_sample]
    langs = [s["language"] for s in ensemble_metrics.per_sample]
    
    lang_metrics = compute_metrics_by_language(refs, hyps, langs)
    print_metrics_table(lang_metrics)
    
    print("\n🏠 FIELD EXTRACTION ACCURACY")
    print("-"*60)
    for field, metrics in sorted(field_metrics.items(), key=lambda x: x[1]["accuracy"], reverse=True):
        print(f"   {field:<25} Acc: {metrics['accuracy']:.4f} ± {metrics['std']:.4f} (n={metrics['count']})")
    
    # Save results
    save_results(output_dir, ensemble_metrics, field_metrics, args.per_image)
    
    # Generate plots
    if args.plots:
        generate_plots(output_dir, ensemble_metrics, lang_metrics)
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()