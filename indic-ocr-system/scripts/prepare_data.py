#!/usr/bin/env python3
"""
Data Preparation Script for IndicOCR
=====================================
Create sample test data and synthetic training data.
"""

import argparse
import os
import json
import random
from pathlib import Path
from typing import List, Dict
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.image_utils import cv_to_pil, pil_to_cv
import cv2


# Sample land record templates for different languages
LAND_RECORD_TEMPLATES = {
    "hi": {
        "fields": {
            "survey_number": "खसरा नंबर: {survey}",
            "khata_number": "खतौनी नंबर: {khata}",
            "village": "मौजा: {village}",
            "tehsil": "तहसील: {tehsil}",
            "district": "जिला: {district}",
            "state": "राज्य: {state}",
            "area": "रकबा: {area} हेक्टेयर",
            "land_type": "भूमि प्रकार: {land_type}",
            "mutation": "नामांतरण नंबर: {mutation}",
            "date": "पंजीकरण दिनांक: {date}"
        },
        "values": {
            "survey": ["142/2A", "256/1B", "789/3C", "101/4D", "202/5E"],
            "khata": ["00345", "00678", "01234", "00999", "00555"],
            "village": ["रामपुर", "श्यामनगर", "गोपालपुर", "कृष्णानगर", "राधानगर"],
            "tehsil": ["सदर", "बख्शी का तालाब", "मोहनलालगंज", "सरोजनी नगर", "मलीहाबाद"],
            "district": ["लखनऊ", "कानपुर", "आगरा", "वाराणसी", "प्रयागराज"],
            "state": ["उत्तर प्रदेश"],
            "area": ["0.2520", "0.5000", "1.2500", "0.7500", "2.0000"],
            "land_type": ["कृषि भूमि", "आवासीय", "वाणिज्यिक", "बंजर", "वन भूमि"],
            "mutation": ["MUT/2024/001234", "MUT/2024/005678", "MUT/2023/009999"],
            "date": ["15/03/2024", "22/01/2024", "10/07/2023", "05/11/2023"]
        }
    },
    "bn": {
        "fields": {
            "survey_number": "দাগ নম্বর: {survey}",
            "khata_number": "খতিয়ান নম্বর: {khata}",
            "village": "মৌজা: {village}",
            "tehsil": "উপজেলা: {tehsil}",
            "district": "জেলা: {district}",
            "state": "রাজ্য: {state}",
            "area": "রকবা: {area} হেক্টেয়ার",
            "land_type": "জমির ধরন: {land_type}",
            "mutation": "নামজারি নম্বর: {mutation}",
            "date": "নিবন্ধন তারিখ: {date}"
        },
        "values": {
            "survey": ["১৪২/২A", "২৫৬/১B", "৭৮৯/৩C"],
            "khata": ["০০৩৪৫", "০০৬৭৮", "০১২৩৪"],
            "village": ["রামপুর", "শ্যামনগর", "গোপালপুর"],
            "tehsil": ["সদর", "বকশীর তালাব", "মোহনলালগঞ্জ"],
            "district": ["লখনৌ", "কানপুর", "আগ্রা"],
            "state": ["উত্তর প্রদেশ"],
            "area": ["০.২৫২০", "০.৫০০০", "১.২৫০০"],
            "land_type": ["কৃষি জমি", "আবাসিক", "বাণিজ্যিক"],
            "mutation": ["MUT/2024/001234", "MUT/2024/005678"],
            "date": ["১৫/০৩/২০২৪", "২২/০১/২০২৪"]
        }
    },
    "ta": {
        "fields": {
            "survey_number": "தொடர்பு எண்: {survey}",
            "khata_number": "பட்டா எண்: {khata}",
            "village": "கிராமம்: {village}",
            "tehsil": "தாலுகா: {tehsil}",
            "district": "மாவட்டம்: {district}",
            "state": "மாநிலம்: {state}",
            "area": "விரிவ்: {area} ஹெக்டேர்",
            "land_type": "நில வகை: {land_type}",
            "mutation": "நாமமாற்று எண்: {mutation}",
            "date": "பதிவு நாள்: {date}"
        },
        "values": {
            "survey": ["142/2A", "256/1B", "789/3C"],
            "khata": ["00345", "00678", "01234"],
            "village": ["ராமபூர்", "சியாம்நகர்", "கோபாலபூர்"],
            "tehsil": ["சதர்", "பக்ஷీர் தாளாப்", "மோகன்லાલ்கஞ்ச்"],
            "district": ["லக்னவ்", "கான்பூர்", "ஆக்ரா"],
            "state": ["உத்தரப் பிரதேசம்"],
            "area": ["0.2520", "0.5000", "1.2500"],
            "land_type": ["விவசாய நிலம்", "அபிவாசி", "வணிக"],
            "mutation": ["MUT/2024/001234", "MUT/2024/005678"],
            "date": ["15/03/2024", "22/01/2024"]
        }
    },
    "en": {
        "fields": {
            "survey_number": "Survey Number: {survey}",
            "khata_number": "Khata Number: {khata}",
            "village": "Village: {village}",
            "tehsil": "Tehsil: {tehsil}",
            "district": "District: {district}",
            "state": "State: {state}",
            "area": "Area: {area} hectare",
            "land_type": "Land Type: {land_type}",
            "mutation": "Mutation Number: {mutation}",
            "date": "Registration Date: {date}"
        },
        "values": {
            "survey": ["142/2A", "256/1B", "789/3C", "101/4D", "202/5E"],
            "khata": ["00345", "00678", "01234", "00999", "00555"],
            "village": ["Rampur", "Shyamnagar", "Gopalpur", "Krishnanagar", "Radhanagar"],
            "tehsil": ["Sadar", "Bakshi Ka Talab", "Mohanlalganj", "Sarojini Nagar", "Malihabad"],
            "district": ["Lucknow", "Kanpur", "Agra", "Varanasi", "Prayagraj"],
            "state": ["Uttar Pradesh"],
            "area": ["0.2520", "0.5000", "1.2500", "0.7500", "2.0000"],
            "land_type": ["Agricultural", "Residential", "Commercial", "Barren", "Forest"],
            "mutation": ["MUT/2024/001234", "MUT/2024/005678", "MUT/2023/009999"],
            "date": ["15/03/2024", "22/01/2024", "10/07/2023", "05/11/2023"]
        }
    }
}


def generate_land_record_text(lang: str) -> str:
    """Generate a land record text for given language"""
    template = LAND_RECORD_TEMPLATES.get(lang, LAND_RECORD_TEMPLATES["en"])
    
    lines = []
    for field, fmt in template["fields"].items():
        value = random.choice(template["values"][field])
        lines.append(fmt.format(**{field: value}))
    
    return "\n".join(lines)


def create_synthetic_image(text: str, lang: str, size: tuple = (800, 600)) -> np.ndarray:
    """Create synthetic image with text"""
    # Create blank image
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 255
    
    # Convert to PIL for text rendering
    pil_img = cv_to_pil(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Try to load a font
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 20)
                break
            except:
                pass
    
    if font is None:
        font = ImageFont.load_default()
    
    # Draw text lines
    y = 50
    for line in text.split("\n"):
        draw.text((50, y), line, fill=(0, 0, 0), font=font)
        y += 35
    
    # Add some noise
    noise = np.random.normal(0, 5, pil_img.size[::-1] + (3,))
    pil_img = Image.fromarray(np.clip(np.array(pil_img) + noise, 0, 255).astype(np.uint8))
    
    # Convert back to OpenCV
    return pil_to_cv(pil_img)


def create_test_data(args):
    """Create test dataset"""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    languages = args.languages or ["hi", "en", "bn", "ta"]
    num_samples = args.num_samples
    
    print(f"📝 Generating {num_samples} samples per language...")
    
    all_annotations = []
    
    for lang in languages:
        lang_dir = output_dir / lang
        lang_dir.mkdir(exist_ok=True)
        images_dir = lang_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        for i in range(num_samples):
            # Generate text
            text = generate_land_record_text(lang)
            
            # Create image
            img = create_synthetic_image(text, lang)
            
            # Save image
            img_name = f"{lang}_land_record_{i:04d}.jpg"
            img_path = images_dir / img_name
            cv2.imwrite(str(img_path), img)
            
            # Save ground truth text
            txt_path = lang_dir / f"{lang}_land_record_{i:04d}.txt"
            txt_path.write_text(text, encoding="utf-8")
            
            # Save language tag
            lang_path = lang_dir / f"{lang}_land_record_{i:04d}.lang"
            lang_path.write_text(lang)
            
            # Save structured ground truth
            json_data = {
                "full_text": text,
                "language": lang,
                "fields": {}
            }
            
            # Parse fields from text
            for line in text.split("\n"):
                if ": " in line:
                    key, value = line.split(": ", 1)
                    json_data["fields"][key.lower().replace(" ", "_")] = {
                        "value": value,
                        "confidence": 1.0
                    }
            
            json_path = lang_dir / f"{lang}_land_record_{i:04d}.json"
            json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2))
            
            all_annotations.append({
                "file_name": img_name,
                "text": text,
                "language": lang
            })
    
    # Save COCO-style annotations
    annotations = {
        "images": [{"file_name": a["file_name"], "id": i} for i, a in enumerate(all_annotations)],
        "annotations": [{"id": i, "image_id": i, "text": a["text"], "language": a["language"]} 
                       for i, a in enumerate(all_annotations)],
        "categories": [{"id": i, "name": lang} for i, lang in enumerate(languages)]
    }
    
    ann_file = output_dir / "annotations.json"
    ann_file.write_text(json.dumps(annotations, ensure_ascii=False, indent=2))
    
    print(f"✅ Generated test data in: {output_dir}")
    print(f"   Languages: {languages}")
    print(f"   Samples per language: {num_samples}")
    print(f"   Total images: {len(all_annotations)}")


def create_benchmark_splits(args):
    """Split data into train/val/test"""
    import shutil
    from sklearn.model_selection import train_test_split
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all images
    all_files = []
    for lang_dir in data_dir.iterdir():
        if lang_dir.is_dir():
            images_dir = lang_dir / "images"
            if images_dir.exists():
                for img in images_dir.glob("*.jpg"):
                    all_files.append(img)
    
    # Split
    train_files, temp_files = train_test_split(all_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
    # Copy to splits
    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        for img_path in files:
            # Copy image
            dst_img = split_dir / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Copy associated files
            for suffix in [".txt", ".lang", ".json"]:
                src = img_path.with_suffix(suffix)
                if src.exists():
                    shutil.copy2(src, split_dir / src.name)
        
        print(f"   {split_name}: {len(files)} images")
    
    print(f"✅ Splits created in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare data for IndicOCR")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Generate synthetic data
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic land records")
    gen_parser.add_argument("--output-dir", default="data/synthetic", help="Output directory")
    gen_parser.add_argument("--languages", nargs="+", default=["hi", "en", "bn", "ta", "te", "kn", "ml"])
    gen_parser.add_argument("--num-samples", type=int, default=100, help="Samples per language")
    
    # Create splits
    split_parser = subparsers.add_parser("split", help="Create train/val/test splits")
    split_parser.add_argument("--data-dir", required=True, help="Source data directory")
    split_parser.add_argument("--output-dir", default="data/splits", help="Output directory")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        create_test_data(args)
    elif args.command == "split":
        create_benchmark_splits(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()