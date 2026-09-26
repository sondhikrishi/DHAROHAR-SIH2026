# Contributing to IndicOCR

Thank you for your interest in contributing to IndicOCR! This project was built for the SIH (Smart India Hackathon) and we welcome contributions from the community.

## 🎯 Ways to Contribute

### 1. **Add New Language Support**
- Add language configuration in `configs/language_mapping.yaml`
- Add vocabulary in `configs/land_record_schema.yaml`
- Test with sample images

### 2. **Improve Model Accuracy**
- Fine-tune TrOCR on domain-specific data
- Add custom PaddleOCR models
- Improve post-processing rules

### 3. **Bug Fixes & Optimizations**
- Fix OCR errors for specific document types
- Improve preprocessing for degraded documents
- Optimize inference speed

### 4. **Documentation & Examples**
- Add more usage examples
- Improve API documentation
- Create tutorials

### 5. **New Features**
- Support for new document types
- Additional export formats
- Integration with other tools

## 🚀 Getting Started

### 1. Fork & Clone
```bash
git clone https://github.com/yourteam/indic-ocr-system.git
cd indic-ocr-system
```

### 2. Setup Development Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

### 3. Run Tests
```bash
pytest tests/ -v
```

### 4. Code Style
```bash
# Format code
black src/ examples/ scripts/ tests/

# Lint
ruff check src/ examples/ scripts/ tests/

# Type check
mypy src/
```

## 📝 Pull Request Process

### 1. Create Feature Branch
```bash
git checkout -b feat/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 2. Make Changes
- Write clean, documented code
- Add tests for new functionality
- Update documentation if needed

### 3. Commit with Convention
```bash
git commit -m "feat: add Gujarati language support with custom vocabulary"
git commit -m "fix: correct Hindi matra confusion in post-processing"
git commit -m "docs: add API usage examples"
```

### 4. Push & Create PR
```bash
git push origin feat/your-feature-name
# Create PR on GitHub
```

## 📋 Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Code style (formatting, etc.) |
| `refactor` | Code refactoring |
| `test` | Adding tests |
| `chore` | Maintenance tasks |

Examples:
```
feat: add Odia language support
fix: resolve Tesseract Urdu recognition issue
docs: update README with deployment guide
refactor: optimize ensemble voting logic
test: add benchmark for Malayalam land records
```

## 🧪 Testing Guidelines

### Unit Tests
```python
# tests/test_recognizers.py
def test_paddle_recognizer_hindi():
    recognizer = PaddleRecognizer(config)
    recognizer.initialize()
    
    # Test with sample
    result = recognizer.recognize(image, regions)
    assert len(result) > 0
    assert all(r.confidence > 0 for r in result)
```

### Integration Tests
```python
# tests/test_pipeline.py
def test_full_pipeline_hindi_land_record():
    ocr = create_ocr(languages=["hi"])
    result = ocr.process("tests/data/hi_land_record.jpg")
    
    assert result.detected_language == LanguageCode.HI
    assert "खसरा" in result.full_text or "khasra" in result.full_text.lower()
    assert result.structured_data is not None
```

### Benchmark Tests
```bash
# Run full benchmark
python examples/benchmark.py data/test --output benchmark_results/
```

## 🌐 Adding a New Language

### 1. Update Language Mapping (`configs/language_mapping.yaml`)
```yaml
languages:
  xx:  # ISO 639-1 code
    name: "Language Name"
    native_name: "Native Name"
    script: "ScriptName"
    iso639_1: "xx"
    iso639_3: "xxx"
    tesseract_code: "xxx"  # If available
    easyocr_code: "xx"     # If available
    paddle_lang: "indian"
    trocr_supported: true
    region: "Region"
    speakers_millions: 10
    priority: 3
```

### 2. Add Vocabulary (`configs/land_record_schema.yaml`)
```yaml
land_record_vocab:
  xx:
    keywords:
      - "keyword1"
      - "keyword2"
```

### 3. Add Tesseract Traineddata
Place `xxx.traineddata` in `models/tesseract/tessdata/`

### 4. Test
```bash
python examples/demo.py --image test_xx.jpg --languages xx
python examples/benchmark.py --lang xx
```

## 🐛 Reporting Issues

### Bug Report Template
```markdown
**Bug Description**
Clear description of the issue

**Steps to Reproduce**
1. Step 1
2. Step 2
3. Step 3

**Expected Behavior**
What should happen

**Actual Behavior**
What actually happens

**Environment**
- OS: Ubuntu 22.04 / Windows 11 / macOS
- Python: 3.10
- GPU: RTX 3090 / CPU only
- Models: PaddleOCR, TrOCR, etc.

**Sample Image** (if applicable)
Attach or link to sample image

**Logs/Error Messages**
```
Paste relevant logs here
```
```

### Feature Request Template
```markdown
**Feature Description**
Clear description of the feature

**Use Case**
Why is this needed?

**Proposed Solution**
How should it work?

**Alternatives Considered**
Other approaches considered
```

## 📚 Code Style Guide

### Python Style
- Follow PEP 8
- Use type hints
- Max line length: 100
- Use `black` for formatting
- Use `ruff` for linting

### Documentation
- Docstrings for all public functions/classes
- Google-style docstrings
- Update README for new features

### Configuration
- Use YAML for config files
- Keep configs in `configs/`
- Document all config options

## 🏷️ Release Process

1. Update version in `pyproject.toml` and `src/__init__.py`
2. Update `CHANGELOG.md`
3. Create release tag: `git tag v1.1.0`
4. Push tag: `git push origin v1.1.0`
5. GitHub Actions will build and publish

## 🤝 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- No harassment or discrimination

## 📞 Getting Help

- **GitHub Discussions**: For questions and ideas
- **GitHub Issues**: For bugs and feature requests
- **Email**: team@sih-hackathon.com

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Annual contributor spotlight

---

**Thank you for contributing to IndicOCR!** 🇮🇳

Together, we're making Indian language OCR accessible to everyone.