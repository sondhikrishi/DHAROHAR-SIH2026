import os

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_enable_pir_api"] = "0"

import paddle

paddle.set_flags({
    "FLAGS_use_mkldnn": False,
    "FLAGS_enable_pir_api": False
})

from paddleocr import PaddleOCR
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

    try:
        results = ocr.predict(
            image_path
        )

        texts = []
        scores = []

        for result in results:

            result_texts = result.get("rec_texts", [])
            result_scores = result.get("rec_scores", [])

            texts.extend(result_texts)
            scores.extend(result_scores)

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

    except Exception as e:
        raise RuntimeError(
            f"PaddleOCR prediction failed: {str(e)}"
        ) from e