from __future__ import annotations

import tempfile
from pathlib import Path

import requests

from src.api.documents import write_docx_file, write_text_file


DEFAULT_API_URL = "http://127.0.0.1:8000"


def _read_uploaded_bytes(image_path: str | Path) -> bytes:
    if not image_path:
        raise ValueError("Please upload an image.")
    return Path(image_path).read_bytes()


def _post_ocr(api_url: str, image_path: str | Path) -> str:
    payload = _read_uploaded_bytes(image_path)
    response = requests.post(
        f"{api_url.rstrip('/')}/ocr",
        files={"image": (Path(image_path).name, payload)},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["ocr_text"]


def _convert_to_text(api_url: str, image_path: str | Path):
    text = _post_ocr(api_url, image_path)
    temp_dir = Path(tempfile.mkdtemp(prefix="arabic_ocr_"))
    txt_path = write_text_file(text, temp_dir / "arabic_ocr.txt")
    docx_path = write_docx_file(text, temp_dir / "arabic_ocr.docx")
    return text, str(txt_path), str(docx_path)


def _query_text(api_url: str, question: str):
    question = (question or "").strip()
    if not question:
        raise ValueError("Please enter a question.")
    response = requests.post(
        f"{api_url.rstrip('/')}/query",
        json={"question": question},
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    return data["answer"], data.get("sources", [])


def _query_image(api_url: str, image_path: str | Path, question: str):
    question = (question or "").strip()
    if not question:
        raise ValueError("Please enter a question.")
    payload = _read_uploaded_bytes(image_path)
    response = requests.post(
        f"{api_url.rstrip('/')}/query-image",
        data={"question": question},
        files={"image": (Path(image_path).name, payload)},
        timeout=300,
    )
    response.raise_for_status()
    data = response.json()
    return data["ocr_text"], data["answer"], data.get("sources", [])


def build_app(api_url: str = DEFAULT_API_URL):
    """Build the two-tab Gradio UI; the API remains the OCR/RAG owner."""
    import gradio as gr

    with gr.Blocks(title="Arabic Handwritten Document Assistant") as app:
        gr.Markdown("# Arabic Handwritten Document Assistant")
        gr.Markdown("Convert handwritten Arabic documents to text or ask questions.")
        with gr.Tab("Convert Handwriting to Text"):
            ocr_image = gr.Image(
                type="filepath",
                label="Arabic handwritten document image",
            )
            convert_button = gr.Button("Convert to Digital Text", variant="primary")
            ocr_output = gr.Textbox(
                label="Extracted Arabic digital text",
                lines=16,
                rtl=True,
            )
            txt_download = gr.File(label="Download TXT")
            docx_download = gr.File(label="Download DOCX")
            convert_button.click(
                _convert_to_text,
                inputs=[gr.State(api_url), ocr_image],
                outputs=[ocr_output, txt_download, docx_download],
            )

        with gr.Tab("Ask Questions"):
            text_question = gr.Textbox(label="Question", rtl=True)
            text_button = gr.Button("Ask about indexed documents")
            text_answer = gr.Textbox(label="Answer", lines=8, rtl=True)
            text_sources = gr.JSON(label="Sources")
            text_button.click(
                _query_text,
                inputs=[gr.State(api_url), text_question],
                outputs=[text_answer, text_sources],
            )

            gr.Markdown("### Ask about a handwritten image")
            image_question = gr.Textbox(label="Question about image", rtl=True)
            question_image = gr.Image(type="filepath", label="Document image")
            image_button = gr.Button("OCR image and answer", variant="primary")
            image_ocr = gr.Textbox(label="OCR preview", lines=10, rtl=True)
            image_answer = gr.Textbox(label="Answer", lines=8, rtl=True)
            image_sources = gr.JSON(label="Sources")
            image_button.click(
                _query_image,
                inputs=[gr.State(api_url), question_image, image_question],
                outputs=[image_ocr, image_answer, image_sources],
            )
    return app


if __name__ == "__main__":
    build_app().launch()
