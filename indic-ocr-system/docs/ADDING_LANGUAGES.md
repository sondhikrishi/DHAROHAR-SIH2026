# Adding New Language Support

This guide explains how to add a new Indian language to IndicOCR.

## Overview

Adding a language involves:
1. Language configuration
2. Vocabulary for land records
3. Tesseract traineddata (if available)
4. Testing and benchmarking

## Step 1: Add Language Definition

Edit `configs/language_mapping.yaml`:

```yaml
languages:
  xx:  # ISO 639-1 code (e.g., "ks" for Kashmiri, "sd" for Sindhi)
    name: "Language Name"
    native_name: "Native Script Name"
    script: "ScriptName"  # Must match SCRIPT_RANGES in language_id/__init__.py
    iso639_1: "xx"
    iso639_3: "xxx"
    tesseract_code: "xxx"  # Tesseract language code (if available)
    easyocr_code: "xx"     # EasyOCR code (if available)
    paddle_lang: "indian"  # PaddleOCR language group
    trocr_supported: true  # Whether TrOCR can be fine-tuned
    region: "Region/State"
    speakers_millions: 10
    priority: 3  # 1=high, 2=medium, 3=low
```

### Script Requirements

The `script` must be one of the predefined scripts in `src/language_id/__init__.py`:
- `Devanagari`, `Bengali`, `Oriya`, `Tamil`, `Telugu`, `Kannada`, `Malayalam`
- `Gujarati`, `Gurmukhi`, `Arabic`, `Latin`

If your language uses a new script, add it to `SCRIPT_RANGES` in `src/language_id/__init__.py`:

```python
SCRIPT_RANGES = {
    # ... existing ...
    ScriptType.NEW_SCRIPT: [(0xXXXX, 0xXXXX)],  # Unicode range
}
```

And add mapping:
```python
LANGUAGE_SCRIPT[LanguageCode.XX] = ScriptType.NEW_SCRIPT
SCRIPT_LANGUAGES[ScriptType.NEW_SCRIPT] = [LanguageCode.XX]
```

## Step 2: Add Land Record Vocabulary

Edit `configs/land_record_schema.yaml`, add to `land_record_vocab`:

```yaml
land_record_vocab:
  xx:
    keywords:
      - "keyword1"      # Survey number
      - "keyword2"      # Khata/Khasra
      - "keyword3"      # Village
      - "keyword4"      # Tehsil
      - "keyword5"      # District
      - "keyword6"      # State
      - "keyword7"      # Area
      - "keyword8"      # Land type
      - "keyword9"      # Mutation
      - "keyword10"     # Registration
      - "keyword11"     # Buyer/Seller
      - "keyword12"     # Units (hectare, acre, etc.)
```

Use the native script. These are used for:
- Vocabulary-based correction
- Field extraction confidence boosting
- Keyword-based language disambiguation

## Step 3: Add Field Extraction Patterns

In `configs/land_record_schema.yaml`, under `fields`, add patterns for the new language:

```yaml
fields:
  survey_number:
    patterns:
      - regex: "(?:local_term1|local_term2)[\\s:]*([A-Z0-9/\\-]+)"
        languages: ["xx", "en"]
        group: 1
    # ... add for other fields
```

Each field needs language-specific regex patterns. Key fields:
- `survey_number`, `khata_number`, `khasra_number`
- `owner_name`, `father_name`
- `village`, `tehsil`, `district`, `state`
- `area`, `land_type`
- `mutation_number`, `registration_date`
- `stamp_duty`, `seller`, `buyer`

## Step 4: Add Tesseract Traineddata (if available)

1. Download `.traineddata` file from [tessdata_best](https://github.com/tesseract-ocr/tessdata_best)
2. Place in `models/tesseract/tessdata/{code}.traineddata`
3. Add language code to `tesseract.langs` in `configs/model_config.yaml`

If no traineddata exists, Tesseract will skip this language.

## Step 5: Add EasyOCR Support (if available)

Check [EasyOCR supported languages](https://github.com/JaidedAI/EasyOCR/blob/master/easyocr/config.py).
If supported, add to `easyocr.langs` in `configs/model_config.yaml`.

## Step 6: Add Language Keywords for Disambiguation

In `src/language_id/__init__.py`, add to `language_keywords`:

```python
self.language_keywords = {
    # ... existing ...
    LanguageCode.XX: ["keyword1", "keyword2", "keyword3", ...],
}
```

These are common function words (postpositions, auxiliaries, common verbs) that distinguish this language from others sharing the same script.

## Step 7: Test the Language

### Quick Test
```bash
python examples/demo.py --image test_xx.jpg --languages xx en
```

### Full Benchmark
```bash
# Prepare test data
mkdir -p data/test/xx
# Add test images with .txt ground truth

python examples/benchmark.py data/test --output benchmark_xx/
```

### Verify Components
```python
from indic_ocr import create_ocr
from src.language_id import LanguageIdentifier

# Test language ID
lid = LanguageIdentifier({})
lid.initialize()
lang, conf = lid.identify("your test text in new language")
print(f"Detected: {lang.value} (conf: {conf:.2f})")

# Test full pipeline
ocr = create_ocr(languages=["xx", "en"])
result = ocr.process("test_image.jpg")
print(f"Language: {result.detected_language.value}")
print(f"Text: {result.full_text}")
```

## Step 8: Benchmark and Document

Run benchmark and record results:

| Language | PaddleOCR | TrOCR | Tesseract | Ensemble |
|----------|-----------|-------|-----------|----------|
| XX (new) | XX% | XX% | XX% | **XX%** |

Add to README.md language table.

## Common Issues

### Language Not Detected
- Check script detection: `identifier.get_script(text)`
- Verify Unicode ranges in `SCRIPT_RANGES`
- Add more keywords to `language_keywords`

### Poor OCR Accuracy
- Add more vocabulary to `land_record_vocab`
- Fine-tune TrOCR on language-specific data
- Check Tesseract traineddata quality

### Field Extraction Fails
- Verify regex patterns match actual text format
- Test patterns: `python -c "import re; print(re.findall(pattern, text))"`
- Add language to pattern's `languages` list

### Missing Model Support
- PaddleOCR: Uses "indian" model (covers 12 languages)
- TrOCR: Requires fine-tuning for new languages
- Tesseract: Needs traineddata file
- EasyOCR: Limited to supported languages

## Example: Adding Kashmiri (ks)

```yaml
# language_mapping.yaml
ks:
  name: "Kashmiri"
  native_name: "कॉशुर / كٲشُر"
  script: "Devanagari"  # or Arabic for Perso-Arabic script
  iso639_1: "ks"
  iso639_3: "kas"
  tesseract_code: "kas"
  easyocr_code: ""
  paddle_lang: "indian"
  trocr_supported: true
  region: "Jammu & Kashmir"
  speakers_millions: 7
  priority: 3
```

```yaml
# land_record_schema.yaml - land_record_vocab
ks:
  keywords:
    - "खसरा"      # survey (Devanagari)
    - "कशूर"      # Kashmir
    - "गाव"       # village
    - "तहसील"    # tehsil
    # ... etc
```

## Checklist

- [ ] Language definition in `language_mapping.yaml`
- [ ] Script support in `language_id/__init__.py`
- [ ] Vocabulary in `land_record_schema.yaml`
- [ ] Field patterns in `land_record_schema.yaml`
- [ ] Keywords for disambiguation
- [ ] Tesseract traineddata (if available)
- [ ] EasyOCR support (if available)
- [ ] Test images with ground truth
- [ ] Benchmark results recorded
- [ ] Documentation updated

## Contributing Back

Once tested, submit a PR with:
1. Language configuration changes
2. Sample test images (5-10)
3. Benchmark results
4. Updated documentation

This helps the community benefit from your work!