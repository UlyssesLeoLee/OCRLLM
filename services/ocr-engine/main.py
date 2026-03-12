"""
OCRLLM OCR Engine - Paragraph-level PDF Extraction
====================================================
Extracts text from PDFs using pdfplumber. Groups words into
paragraph-level TextBlocks suitable for semantic embedding.

Tracks:
  A - Digital PDF: pdfplumber text extraction with paragraph merging
  B - Scanned PDF:  (future) image-based OCR fallback via pytesseract

Port: 8000
"""

import json
import os
import uuid
from pathlib import Path

import pdfplumber
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="OCRLLM Python OCR Engine")

# Absolute path to the project root (two levels up from this file: services/ocr-engine/ → root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIFIED_JSON_DIR = PROJECT_ROOT / "storage" / "unified_jsons"
UNIFIED_JSON_DIR.mkdir(parents=True, exist_ok=True)


class ExtractionRequest(BaseModel):
    task_id: str
    pdf_path: str
    primary_language: str


def _words_to_paragraphs(words: list[dict], line_gap_threshold: float = 5.0, para_gap_threshold: float = 14.0) -> list[str]:
    """
    Merge pdfplumber word-level dicts into paragraph strings.

    Algorithm:
    1. Sort words by their vertical midpoint (top) then x0.
    2. Group words onto the same "line" when their y-gap < line_gap_threshold.
    3. Merge adjacent lines into a "paragraph" when line-to-line gap < para_gap_threshold.
    4. Return one string per paragraph.
    """
    if not words:
        return []

    # Sort: primarily by vertical position, secondarily by horizontal
    words = sorted(words, key=lambda w: (round(w["top"] / line_gap_threshold), w["x0"]))

    lines: list[dict] = []  # each: {text, bottom}
    current_line_words: list[str] = []
    current_line_bottom: float = words[0]["top"]

    for word in words:
        if abs(word["top"] - current_line_bottom) <= line_gap_threshold:
            current_line_words.append(word["text"])
        else:
            if current_line_words:
                lines.append({"text": " ".join(current_line_words), "bottom": current_line_bottom})
            current_line_words = [word["text"]]
            current_line_bottom = word["top"]

    if current_line_words:
        lines.append({"text": " ".join(current_line_words), "bottom": current_line_bottom})

    # Merge lines into paragraphs
    paragraphs: list[str] = []
    current_para_lines: list[str] = []
    prev_bottom: float | None = None

    for line in lines:
        if prev_bottom is None or (line["bottom"] - prev_bottom) <= para_gap_threshold:
            current_para_lines.append(line["text"])
        else:
            text = "\n".join(current_para_lines).strip()
            if text:
                paragraphs.append(text)
            current_para_lines = [line["text"]]
        prev_bottom = line["bottom"]

    if current_para_lines:
        text = "\n".join(current_para_lines).strip()
        if text:
            paragraphs.append(text)

    return paragraphs


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/v1/extract")
def extract_pdf(req: ExtractionRequest):
    unified_pages = []

    try:
        with pdfplumber.open(req.pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                words = page.extract_words(keep_blank_chars=True)
                paragraphs = _words_to_paragraphs(words)

                blocks = []
                for para_idx, para_text in enumerate(paragraphs):
                    blocks.append({
                        "block_id": f"p{page_idx + 1}_b{para_idx}",
                        "block_type": "text",
                        "text": para_text,
                        # Bounding box is approximate for the whole page paragraph region
                        "bounding_box": [0.0, 0.0, page.width, page.height],
                        "confidence": 1.0,
                        "reading_order": para_idx,
                        "lang_detected": req.primary_language,
                        "metadata": {},
                    })

                unified_pages.append({
                    "page_num": page_idx + 1,
                    "blocks": blocks,
                })

        output_data = {
            "document_id": req.task_id,
            "document_type": "digital_pdf",
            "page_count": len(unified_pages),
            "meta_params": {
                "primary_language": req.primary_language,
                "fallback_languages": [],
            },
            "pages": unified_pages,
        }

        output_path = UNIFIED_JSON_DIR / f"{req.task_id}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        return {
            "status": "success",
            "message": "Extraction complete",
            "output_path": str(output_path),
            "page_count": len(unified_pages),
            "total_blocks": sum(len(p["blocks"]) for p in unified_pages),
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
