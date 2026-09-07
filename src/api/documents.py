from __future__ import annotations

from pathlib import Path

from docx import Document


def write_text_file(text: str, output_path: str | Path) -> Path:
    """Write OCR text as UTF-8 and return the created path."""
    path = Path(output_path)
    path.write_text(text, encoding="utf-8")
    return path


def write_docx_file(text: str, output_path: str | Path) -> Path:
    """Create a readable right-to-left Arabic OCR document."""
    path = Path(output_path)
    document = Document()
    document.add_heading("Arabic Handwritten Document - OCR", level=1)
    for paragraph_text in text.splitlines() or [""]:
        paragraph = document.add_paragraph(paragraph_text)
        paragraph.alignment = 2
        paragraph.paragraph_format.space_after = 0
    document.save(path)
    return path
