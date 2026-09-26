#!/usr/bin/env python3
"""
TrOCR Fine-tuning Script for Indic Languages
=============================================
Fine-tune TrOCR on Indian land record data.
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
    default_data_collator
)
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class IndicOCRDataset(Dataset):
    """Dataset for TrOCR fine-tuning on Indic land records"""
    
    def __init__(
        self, 
        data_dir: Path, 
        processor: TrOCRProcessor,
        max_length: int = 256,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_length = max_length
        self.augment = augment
        
        # Load annotations
        self.samples = self._load_annotations()
        
        print(f"📊 Loaded {len(self.samples)} samples from {data_dir}")
    
    def _load_annotations(self) -> List[Dict]:
        """Load image-text pairs"""
        samples = []
        
        # Support multiple annotation formats
        # Format 1: COCO-style annotations.json
        ann_file = self.data_dir / "annotations.json"
        if ann_file.exists():
            with open(ann_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for ann in data.get("annotations", []):
                img_path = self.data_dir / "images" / ann["file_name"]
                if img_path.exists():
                    samples.append({
                        "image_path": str(img_path),
                        "text": ann["text"],
                        "language": ann.get("language", "hi")
                    })
            return samples
        
        # Format 2: Image + .txt pairs
        extensions = [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]
        for ext in extensions:
            for img_file in self.data_dir.glob(f"*{ext}"):
                txt_file = img_file.with_suffix(".txt")
                if txt_file.exists():
                    text = txt_file.read_text(encoding="utf-8").strip()
                    if text:
                        # Try to get language from .lang file or parent dir
                        lang_file = img_file.with_suffix(".lang")
                        lang = "hi"
                        if lang_file.exists():
                            lang = lang_file.read_text().strip()
                        elif img_file.parent.name in ["hi", "bn", "ta", "te", "kn", "ml", "mr", "gu", "pa", "or", "as", "ur", "en"]:
                            lang = img_file.parent.name
                        
                        samples.append({
                            "image_path": str(img_file),
                            "text": text,
                            "language": lang
                        })
        
        # Format 3: JSONL file
        jsonl_file = self.data_dir / "data.jsonl"
        if jsonl_file.exists():
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    img_path = self.data_dir / item["image"]
                    if img_path.exists():
                        samples.append({
                            "image_path": str(img_path),
                            "text": item["text"],
                            "language": item.get("language", "hi")
                        })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            print(f"⚠️  Failed to load {sample['image_path']}: {e}")
            # Return dummy sample
            image = Image.new("RGB", (384, 384), color="white")
            text = ""
        else:
            text = sample["text"]
        
        # Process image
        pixel_values = self.processor(
            images=image, 
            return_tensors="pt"
        ).pixel_values.squeeze(0)
        
        # Process text
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        # Replace padding token id with -100 (ignored in loss)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


def compute_cer(pred_ids, label_ids, tokenizer):
    """Compute Character Error Rate"""
    from src.utils.metrics import calculate_cer
    
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    cers = [calculate_cer(l, p) for l, p in zip(label_str, pred_str)]
    return np.mean(cers)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune TrOCR on Indic land record data"
    )
    
    parser.add_argument("--data-dir", required=True, help="Training data directory")
    parser.add_argument("--val-dir", help="Validation data directory")
    parser.add_argument("--output-dir", default="models/trocr-indic-finetuned", help="Output directory")
    
    parser.add_argument("--base-model", default="microsoft/trocr-base-handwritten", help="Base model")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length")
    
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision")
    parser.add_argument("--gradient-accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--logging-steps", type=int, default=50, help="Logging interval")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--save-steps", type=int, default=500, help="Save interval")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Load processor and model
    print(f"📥 Loading base model: {args.base_model}")
    processor = TrOCRProcessor.from_pretrained(args.base_model)
    model = VisionEncoderDecoderModel.from_pretrained(args.base_model)
    
    # Configure model
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = args.max_length
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    model.to(device)
    
    # Load datasets
    print("📂 Loading datasets...")
    train_dataset = IndicOCRDataset(
        Path(args.data_dir), 
        processor, 
        max_length=args.max_length,
        augment=True
    )
    
    val_dataset = None
    if args.val_dir:
        val_dataset = IndicOCRDataset(
            Path(args.val_dir),
            processor,
            max_length=args.max_length,
            augment=False
        )
    
    if len(train_dataset) == 0:
        print("❌ No training data found!")
        sys.exit(1)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        fp16=args.fp16 and device == "cuda",
        gradient_accumulation_steps=args.gradient_accumulation,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=args.eval_steps if val_dataset else None,
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="cer" if val_dataset else None,
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        generation_num_beams=4,
        report_to="none",  # Disable wandb/tensorboard
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
    )
    
    # Data collator
    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}
    
    # Compute metrics function
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 with pad token id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        cer = compute_cer(pred_ids, label_ids, processor.tokenizer)
        return {"cer": cer}
    
    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics if val_dataset else None,
    )
    
    # Train
    print("🚀 Starting training...")
    trainer.train()
    
    # Save final model
    print(f"💾 Saving model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    
    # Save training config
    config = {
        "base_model": args.base_model,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset else 0,
    }
    
    with open(Path(args.output_dir) / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Fine-tuning complete!")
    print(f"   Model saved to: {args.output_dir}")
    print(f"   To use: update model_config.yaml with fine_tuned_path: \"{args.output_dir}\"")


if __name__ == "__main__":
    main()