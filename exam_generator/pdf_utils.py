# exam_generator/pdf_utils.py - PDF text extraction, page rendering, and OCR helpers
import io
import base64
from typing import List, Optional

import pdfplumber
from pdf2image import convert_from_bytes

try:
    import pytesseract
    OCR_OK = True
except Exception:
    OCR_OK = False


class PdfMixin:
    """PDF text extraction, page-to-image rendering, and OCR fallback."""

    def extract_text_from_pdf(self, file_bytes: bytes) -> str:
        """Extract text content from PDF past paper"""
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                return "\n\n".join(text_parts)
        except Exception as e:
            print(f"PDF extraction error: {e}")
            return ""

    def _b64_png(self, pil_image) -> str:
        import io, base64
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def pdf_pages_to_images(self, file_bytes: bytes, pages: Optional[List[int]] = None) -> List[str]:
        """Return a list of base64 PNG strings for requested pages (1-based)."""
        try:
            images = convert_from_bytes(file_bytes, fmt="png")
            if pages:
                idxs = [p-1 for p in pages if 1 <= p <= len(images)]
                images = [images[i] for i in idxs]
            return [self._b64_png(im) for im in images]
        except Exception as e:
            print(f"PDF->image error: {e}")
            return []

    def ocr_pdf_text(self, file_bytes: bytes, pages: Optional[List[int]] = None) -> str:
        """OCR fallback when pdfplumber text is empty or partial."""
        if not OCR_OK:
            return ""
        try:
            images = convert_from_bytes(file_bytes, fmt="png")
            if pages:
                idxs = [p-1 for p in pages if 1 <= p <= len(images)]
                images = [images[i] for i in idxs]
            out = []
            for im in images:
                try:
                    out.append(pytesseract.image_to_string(im))
                except Exception:
                    pass
            return "\n\n".join([t for t in out if t and t.strip()])
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
