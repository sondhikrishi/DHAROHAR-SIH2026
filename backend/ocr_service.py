import os

# ============================================================
# ENVIRONMENT SETTINGS
# ============================================================

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# IMPORTANT:
# Import torch BEFORE Paddle/PaddleOCR on this Windows setup.
import torch

import paddle

# Disable problematic CPU features
paddle.set_flags({
    "FLAGS_use_mkldnn": False,
    "FLAGS_enable_pir_api": False
})

from paddleocr import PaddleOCR


# ============================================================
# INITIALIZE HINDI OCR
# ============================================================

print("Initializing Hindi OCR...")

ocr = PaddleOCR(
    lang="hi",
    ocr_version="PP-OCRv5",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

print("Hindi OCR initialized successfully.")


# ============================================================
# OCR FUNCTION
# ============================================================

def extract_text(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"OCR input file not found: {image_path}"
        )

    results = list(ocr.predict(image_path))

    if not results:
        return "", 0.0

    result = results[0]

    texts = result.get("rec_texts", [])
    scores = result.get("rec_scores", [])

    extracted_text = "\n".join(
        text.strip()
        for text in texts
        if text and text.strip()
    )

    if scores:
        confidence = sum(scores) / len(scores)
    else:
        confidence = 0.0

    return extracted_text, confidence