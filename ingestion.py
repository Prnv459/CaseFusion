import io
import re
import os
import tempfile
from typing import List

from fastapi import UploadFile, HTTPException
import fitz  # PyMuPDF
import docx2txt
from PIL import Image
import pytesseract
import whisper

from .schemas import EvidenceDoc, PageText

# 0. MODELS: Whisper + (optional) Gemini Vision

whisper_model = whisper.load_model("base", device="cpu")

# Gemini Vision for OCR on screenshots (if API key is present)
try:
    import google.generativeai as genai

    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        vision_model = genai.GenerativeModel("gemini-2.5-flash")
    else:
        vision_model = None
        print(
            "[CaseFusion] WARNING: GOOGLE_API_KEY not set. Using Tesseract OCR for images.")
except Exception as e:
    vision_model = None
    print(
        f"[CaseFusion] WARNING: google.generativeai not available ({e}). Using Tesseract OCR for images.")


# 1. Helper functions

def _postprocess_text(text: str) -> str:
    """
    Basic cleanup for OCR / extracted text.
    """
    # Collapse many blank lines
    text = re.sub(r"(\n\s*){2,}", "\n\n", text)

    # Join lines broken mid-word/sentence
    text = re.sub(r"([a-zA-Z])\n([a-zA-Z])", r"\1 \2", text)

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def _ocr_image_with_gemini(img_bytes: bytes) -> str:
    """
    High-quality OCR using Gemini Vision. Falls back to empty string if
    something goes wrong (caller will then use Tesseract).
    """
    if vision_model is None:
        return ""

    try:
        # Ensure we send JPEG to Gemini
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        jpeg_bytes = buf.getvalue()

        prompt = (
            "You are an OCR engine. Read ALL text in this WhatsApp/chat/email/screenshot "
            "in natural reading order. Return ONLY the raw text, no explanations, no labels."
        )

        response = vision_model.generate_content(
            [
                {"mime_type": "image/jpeg", "data": jpeg_bytes},
                prompt,
            ],
            generation_config={"temperature": 0.0},
        )
        return (response.text or "").strip()
    except Exception as e:
        print(
            f"[CaseFusion] Gemini Vision OCR failed, falling back to Tesseract: {e}")
        return ""


def _detect_evidence_type(text: str) -> str:
    """
    Very simple heuristic classifier for evidence type.
    You can replace/improve with a ML classifier later.
    """
    lower = text.lower()

    # Signs of email (Gmail-style PDFs, headers, etc.)
    if (
        ("to:" in lower and "@" in lower)
        or ("from:" in lower and "@" in lower)
        or "subject:" in lower
        or "gmail - " in lower[:200]
    ):
        return "email"

    # Signs of chat export (WhatsApp / Telegram style)
    if re.search(r'\d{1,2}/\d{1,2}/\d{2,4}.* - .*: ', text):
        return "chat"

    # Signs of bank statements / account docs
    if any(
        kw in lower
        for kw in ["account number", "a/c no", "ifsc", "utr", "txn", "transaction", "statement"]
    ):
        return "bank_document"

    return "generic"

# 2. Main ingestion function


def ingest_file(file: UploadFile) -> EvidenceDoc:
    """
    Ingests any supported file (PDF, DOCX, image, audio) and returns
    a normalized EvidenceDoc with:
      - raw_text (full text)
      - pages (page-wise text)
      - evidence_type (chat/email/bank_document/generic)
    """
    content_type = file.content_type or ""
    file_bytes = file.file.read()
    pages: List[PageText] = []
    raw_text = ""

    try:
        # --- PDF ---
        if "pdf" in content_type:
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")

            # 1) Try text layer first
            for page_num in range(len(pdf_doc)):
                page = pdf_doc.load_page(page_num)
                page_text = page.get_text("text", sort=True) or ""
                pages.append(
                    PageText(page_number=page_num + 1, text=page_text))

            raw_text = "\n\n".join(p.text for p in pages)

            # If very little text, assume scanned PDF -> OCR
            if len(raw_text.strip()) < 100:
                pages = []
                for page_num in range(len(pdf_doc)):
                    page = pdf_doc.load_page(page_num)
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes(
                        "RGB", [pix.width, pix.height], pix.samples)
                    page_text = pytesseract.image_to_string(img)
                    pages.append(
                        PageText(page_number=page_num + 1, text=page_text))
                raw_text = "\n\n".join(p.text for p in pages)

        # DOCX
        elif "wordprocessingml.document" in content_type:
            text = docx2txt.process(io.BytesIO(file_bytes))
            raw_text = text
            pages = [PageText(page_number=1, text=text)]

        # IMAGE (screenshots, photos of chats/emails/docs) ---
        elif "image" in content_type:
            # 1) Try Gemini Vision OCR
            text = _ocr_image_with_gemini(file_bytes)

            # 2) Fallback to Tesseract if Gemini is unavailable or failed
            if not text.strip():
                img = Image.open(io.BytesIO(file_bytes))
                text = pytesseract.image_to_string(img)

            raw_text = text
            pages = [PageText(page_number=1, text=text)]

        # AUDIO (voice notes, call recordings)
        elif "audio" in content_type:
            print(f"Processing audio file: {file.filename}")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                temp_audio.write(file_bytes)
                temp_audio_path = temp_audio.name

            try:
                result = whisper_model.transcribe(temp_audio_path, fp16=False)
                raw_text = result["text"]
                pages = [PageText(page_number=1, text=raw_text)]
                print("Audio transcription complete.")
            finally:
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {content_type}",
            )

        # Post-process text globally
        raw_text = _postprocess_text(raw_text)
        pages = [
            PageText(page_number=p.page_number, text=_postprocess_text(p.text))
            for p in pages
        ]

        evidence_type = _detect_evidence_type(raw_text)

        return EvidenceDoc(
            filename=file.filename,
            content_type=content_type,
            raw_text=raw_text,
            pages=pages,
            evidence_type=evidence_type,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process file: {str(e)}",
        )
