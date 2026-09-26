"""
Metrics and evaluation utilities for IndicOCR
"""

import re
import unicodedata
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from collections import Counter


def normalize_text(text: str, language: str = "en") -> str:
    """
    Normalize text for comparison
    
    Args:
        text: Input text
        language: Language code for language-specific normalization
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Unicode normalization
    text = unicodedata.normalize('NFC', text)
    
    # Remove zero-width characters
    text = re.sub(r'[\u200b-\u200f\ufeff]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip
    text = text.strip()
    
    # Language-specific normalization
    if language in ['hi', 'mr', 'ne', 'bn', 'as', 'or', 'gu', 'pa']:
        # Indic scripts: normalize nukta, etc.
        text = normalize_indic(text, language)
    
    return text


def normalize_indic(text: str, language: str) -> str:
    """Language-specific normalization for Indic scripts"""
    # Common normalizations
    # Remove extra virama/halant at end
    text = re.sub(r'[\u094d\u09cd\u0a4d\u0acd\u0b4d\u0bcd\u0c4d\u0ccd\u0d4d]+$', '', text)
    
    # Normalize chandrabindu/anusvara
    if language in ['hi', 'mr', 'ne']:
        text = text.replace('\u0901', '\u0902')  # Chandrabindu -> Anusvara
    
    return text


def calculate_cer(reference: str, hypothesis: str, language: str = "en") -> float:
    """
    Calculate Character Error Rate (CER)
    
    CER = (Substitutions + Deletions + Insertions) / Total characters in reference
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        language: Language code for normalization
        
    Returns:
        CER score (0.0 = perfect, 1.0 = completely wrong)
    """
    ref = normalize_text(reference, language)
    hyp = normalize_text(hypothesis, language)
    
    if not ref:
        return 0.0 if not hyp else 1.0
    
    # Character-level Levenshtein distance
    distance = levenshtein_distance(list(ref), list(hyp))
    return distance / len(ref)


def calculate_wer(reference: str, hypothesis: str, language: str = "en") -> float:
    """
    Calculate Word Error Rate (WER)
    
    WER = (Substitutions + Deletions + Insertions) / Total words in reference
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        language: Language code for normalization
        
    Returns:
        WER score (0.0 = perfect, 1.0 = completely wrong)
    """
    ref = normalize_text(reference, language)
    hyp = normalize_text(hypothesis, language)
    
    # Split into words (language-aware)
    ref_words = tokenize_words(ref, language)
    hyp_words = tokenize_words(hyp, language)
    
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    
    distance = levenshtein_distance(ref_words, hyp_words)
    return distance / len(ref_words)


def tokenize_words(text: str, language: str) -> List[str]:
    """Tokenize text into words (language-aware)"""
    if language in ['hi', 'mr', 'ne', 'bn', 'as', 'or', 'gu', 'pa', 'ta', 'te', 'kn', 'ml']:
        # For Indic languages, split on whitespace and punctuation
        # This is simplified; for production use indic-nlp-library
        return re.findall(r'\S+', text)
    else:
        # English and other space-separated languages
        return text.split()


def levenshtein_distance(a: List, b: List) -> int:
    """Calculate Levenshtein distance between two sequences"""
    if len(a) < len(b):
        a, b = b, a
    
    if len(b) == 0:
        return len(a)
    
    previous_row = list(range(len(b) + 1))
    
    for i, elem_a in enumerate(a):
        current_row = [i + 1]
        for j, elem_b in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (elem_a != elem_b)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def calculate_f1(reference: str, hypothesis: str, language: str = "en") -> float:
    """
    Calculate F1 score based on character n-grams
    
    Args:
        reference: Ground truth text
        hypothesis: Predicted text
        language: Language code
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    ref = normalize_text(reference, language)
    hyp = normalize_text(hypothesis, language)
    
    if not ref and not hyp:
        return 1.0
    if not ref or not hyp:
        return 0.0
    
    # Character bigrams
    ref_bigrams = get_ngrams(ref, 2)
    hyp_bigrams = get_ngrams(hyp, 2)
    
    if not ref_bigrams and not hyp_bigrams:
        return 1.0
    
    ref_counter = Counter(ref_bigrams)
    hyp_counter = Counter(hyp_bigrams)
    
    # Intersection
    common = sum((ref_counter & hyp_counter).values())
    
    precision = common / sum(hyp_counter.values()) if hyp_counter else 0
    recall = common / sum(ref_counter.values()) if ref_counter else 0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def get_ngrams(text: str, n: int) -> List[str]:
    """Extract character n-grams from text"""
    if len(text) < n:
        return [text] if text else []
    return [text[i:i+n] for i in range(len(text) - n + 1)]


@dataclass
class OCRMetrics:
    """Container for OCR evaluation metrics"""
    cer: float
    wer: float
    f1: float
    accuracy: float  # 1 - CER
    num_samples: int
    per_sample: List[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cer": self.cer,
            "wer": self.wer,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "num_samples": self.num_samples,
            "per_sample": self.per_sample
        }
    
    def summary(self) -> str:
        return (
            f"OCR Metrics Summary:\n"
            f"  Samples: {self.num_samples}\n"
            f"  CER: {self.cer:.4f} ({self.cer*100:.2f}%)\n"
            f"  WER: {self.wer:.4f} ({self.wer*100:.2f}%)\n"
            f"  F1:  {self.f1:.4f} ({self.f1*100:.2f}%)\n"
            f"  Acc: {self.accuracy:.4f} ({self.accuracy*100:.2f}%)"
        )


def compute_metrics(
    references: List[str],
    hypotheses: List[str],
    languages: Optional[List[str]] = None
) -> OCRMetrics:
    """
    Compute aggregate metrics for multiple samples
    
    Args:
        references: List of ground truth texts
        hypotheses: List of predicted texts
        languages: Optional list of language codes per sample
        
    Returns:
        OCRMetrics object
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have same length")
    
    n = len(references)
    if languages is None:
        languages = ["en"] * n
    elif len(languages) != n:
        raise ValueError("Languages list must match samples length")
    
    cers = []
    wers = []
    f1s = []
    per_sample = []
    
    for ref, hyp, lang in zip(references, hypotheses, languages):
        cer = calculate_cer(ref, hyp, lang)
        wer = calculate_wer(ref, hyp, lang)
        f1 = calculate_f1(ref, hyp, lang)
        
        cers.append(cer)
        wers.append(wer)
        f1s.append(f1)
        
        per_sample.append({
            "reference": ref,
            "hypothesis": hyp,
            "language": lang,
            "cer": cer,
            "wer": wer,
            "f1": f1
        })
    
    return OCRMetrics(
        cer=np.mean(cers),
        wer=np.mean(wers),
        f1=np.mean(f1s),
        accuracy=1 - np.mean(cers),
        num_samples=n,
        per_sample=per_sample
    )


def compute_metrics_by_language(
    references: List[str],
    hypotheses: List[str],
    languages: List[str]
) -> Dict[str, OCRMetrics]:
    """Compute metrics grouped by language"""
    lang_groups = {}
    
    for ref, hyp, lang in zip(references, hypotheses, languages):
        if lang not in lang_groups:
            lang_groups[lang] = {"refs": [], "hyps": [], "langs": []}
        lang_groups[lang]["refs"].append(ref)
        lang_groups[lang]["hyps"].append(hyp)
        lang_groups[lang]["langs"].append(lang)
    
    results = {}
    for lang, data in lang_groups.items():
        results[lang] = compute_metrics(data["refs"], data["hyps"], data["langs"])
    
    return results


def print_metrics_table(metrics_by_lang: Dict[str, OCRMetrics]) -> None:
    """Print formatted metrics table"""
    print(f"{'Language':<10} {'Samples':>8} {'CER':>8} {'WER':>8} {'F1':>8} {'Acc':>8}")
    print("-" * 55)
    
    for lang, metrics in sorted(metrics_by_lang.items()):
        print(f"{lang:<10} {metrics.num_samples:>8} {metrics.cer:>8.4f} {metrics.wer:>8.4f} {metrics.f1:>8.4f} {metrics.accuracy:>8.4f}")
    
    # Overall
    all_refs = []
    all_hyps = []
    all_langs = []
    for lang, metrics in metrics_by_lang.items():
        for sample in metrics.per_sample:
            all_refs.append(sample["reference"])
            all_hyps.append(sample["hypothesis"])
            all_langs.append(sample["language"])
    
    overall = compute_metrics(all_refs, all_hyps, all_langs)
    print("-" * 55)
    print(f"{'OVERALL':<10} {overall.num_samples:>8} {overall.cer:>8.4f} {overall.wer:>8.4f} {overall.f1:>8.4f} {overall.accuracy:>8.4f}")


def calculate_field_accuracy(
    ground_truth: Dict[str, str],
    predicted: Dict[str, str],
    fields: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate field-level accuracy for structured extraction
    
    Args:
        ground_truth: Dict of field_name -> value
        predicted: Dict of field_name -> value
        fields: Optional list of fields to evaluate
        
    Returns:
        Dict of field_name -> accuracy
    """
    if fields is None:
        fields = set(ground_truth.keys()) | set(predicted.keys())
    
    results = {}
    for field in fields:
        gt = ground_truth.get(field, "")
        pred = predicted.get(field, "")
        
        if not gt and not pred:
            results[field] = 1.0
        elif not gt or not pred:
            results[field] = 0.0
        else:
            # Use F1 for field comparison
            results[field] = calculate_f1(gt, pred)
    
    return results


def benchmark_models(
    model_results: Dict[str, List[Dict[str, Any]]],
    ground_truth: List[Dict[str, Any]],
    language_field: str = "language"
) -> Dict[str, OCRMetrics]:
    """
    Benchmark multiple models against ground truth
    
    Args:
        model_results: Dict of model_name -> list of predictions per image
        ground_truth: List of ground truth dicts per image
        language_field: Field name for language in ground truth
        
    Returns:
        Dict of model_name -> OCRMetrics
    """
    results = {}
    
    for model_name, predictions in model_results.items():
        references = []
        hypotheses = []
        languages = []
        
        for pred, gt in zip(predictions, ground_truth):
            # Extract full text
            pred_text = " ".join([r.get("text", "") for r in pred.get("regions", [])])
            gt_text = gt.get("full_text", "")
            
            lang = gt.get(language_field, "en")
            
            references.append(gt_text)
            hypotheses.append(pred_text)
            languages.append(lang)
        
        results[model_name] = compute_metrics(references, hypotheses, languages)
    
    return results