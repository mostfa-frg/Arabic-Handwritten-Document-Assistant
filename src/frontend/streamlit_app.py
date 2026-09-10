from __future__ import annotations

import hashlib
import os
from dotenv import load_dotenv

import requests
import streamlit as st


DEFAULT_API_URL = "http://127.0.0.1:8000"
load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", DEFAULT_API_URL).rstrip("/")


def _reset_document_state() -> None:
    st.session_state["ocr_text"] = ""
    st.session_state["edited_text"] = ""
    st.session_state["chat_history"] = []
    st.session_state["txt_bytes"] = None
    st.session_state["docx_bytes"] = None


def _generate_downloads(text: str) -> None:
    """Create downloads from the text the user reviewed, not only raw OCR."""
    for endpoint, key in (("txt", "txt_bytes"), ("docx", "docx_bytes")):
        try:
            response = requests.post(
                f"{API_BASE_URL}/documents/{endpoint}",
                json={"text": text},
                timeout=60,
            )
            if response.ok:
                st.session_state[key] = response.content
            else:
                st.warning(f"{endpoint.upper()} download is unavailable: {_api_error(response)}")
        except requests.RequestException as exc:
            st.warning(f"Could not create {endpoint.upper()} download: {exc}")


def _api_error(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    return str(detail or f"Backend returned HTTP {response.status_code}.")


def _process_image(uploaded_file) -> None:
    image_bytes = uploaded_file.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    _reset_document_state()
    st.session_state["image_bytes"] = image_bytes
    st.session_state["image_name"] = uploaded_file.name
    st.session_state["image_hash"] = image_hash
    try:
        response = requests.post(
            f"{API_BASE_URL}/ocr",
            files={"image": (uploaded_file.name, image_bytes, uploaded_file.type)},
            timeout=300,
        )
        if not response.ok:
            st.error(f"OCR failed: {_api_error(response)}")
            return
        ocr_text = response.json().get("ocr_text", "").strip()
        st.session_state["ocr_text"] = ocr_text
        st.session_state["edited_text"] = ocr_text
        if not ocr_text:
            st.warning("OCR completed, but no text was extracted.")
            return
        _generate_downloads(ocr_text)
        st.success("Image processed successfully.")
    except requests.RequestException as exc:
        st.error(f"Could not connect to FastAPI at {API_BASE_URL}: {exc}")


def _ask_question(question: str) -> None:
    ocr_text = (
        st.session_state.get("edited_text")
        or st.session_state.get("ocr_text", "")
    ).strip()
    if not ocr_text:
        st.warning("Process an image with readable OCR text before asking a question.")
        return
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question, "ocr_text": ocr_text},
            timeout=300,
        )
        if not response.ok:
            st.error(f"Question failed: {_api_error(response)}")
            return
        data = response.json()
        st.session_state["chat_history"].append(
            {
                "question": question,
                "answer": data.get("answer", ""),
                "sources": data.get("sources", []),
            }
        )
    except requests.RequestException as exc:
        st.error(f"Could not connect to FastAPI at {API_BASE_URL}: {exc}")


def _initialize_state() -> None:
    defaults = {
        "ocr_text": "",
        "edited_text": "",
        "chat_history": [],
        "image_bytes": None,
        "image_name": "",
        "image_hash": "",
        "txt_bytes": None,
        "docx_bytes": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def build_app() -> None:
    st.set_page_config(
        page_title="Arabic Document Assistant",
        page_icon="📜",
        layout="wide",
    )
    _initialize_state()
    st.markdown(
        """
        <style>
        .block-container { max-width: 1400px; padding-top: 2rem; }
        [data-testid="stChatMessage"] {
            border-radius: 16px;
            margin: 0.65rem 0;
            padding: 0.7rem 0.9rem;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            min-height: 520px;
            padding: 1.25rem;
            border: 1px solid #35445f;
            border-radius: 20px;
            background: #171c28;
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.18);
        }
        [data-testid="stChatInput"] {
            margin-top: 1.5rem;
        }
        [data-testid="stChatInput"] > div {
            border: 1px solid #62708a;
            border-radius: 14px;
            background: #252d3d;
        }
        [data-testid="stChatInput"] > div:focus-within {
            border-color: #668cff;
            box-shadow: 0 0 0 1px #668cff;
        }
        [data-testid="stChatInput"] textarea {
            color: #f4f7ff;
            caret-color: #8ea8ff;
        }
        [data-testid="stChatInput"] textarea::placeholder {
            color: #c0c9da;
            opacity: 1;
        }
        [data-testid="stChatInput"] button {
            color: #ffffff;
            background: #536dce;
            border-radius: 10px;
        }
        [data-testid="stChatInput"] button:hover {
            background: #6683e8;
        }
        .assistant-empty-state {
            min-height: 300px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            color: #e5eaf5;
        }
        .assistant-empty-avatar {
            font-size: 3.5rem;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Arabic Document Assistant")
    st.markdown(
    "<p style='font-size: 22px;'>.حوّل الوثائق العربية المكتوبة بخط اليد إلى نص، ثم اسأل عنها</p>",
    unsafe_allow_html=True,
     )

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        st.subheader("Document")
        uploaded_file = st.file_uploader(
            "Upload one handwritten Arabic document image",
            type=["png", "jpg", "jpeg", "webp", "tif", "tiff"],
        )
        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            st.image(image_bytes, caption=uploaded_file.name, use_container_width=True)
            if st.button("Process Image", type="primary", use_container_width=True):
                with st.status("Processing OCR...", expanded=False):
                    _process_image(uploaded_file)

        ocr_text = (
            st.session_state.get("edited_text")
            or st.session_state["ocr_text"]
        ).strip()
        if ocr_text:
            st.subheader("Extracted OCR text")
            st.caption("راجع النص وعدّل الكلمات غير الصحيحة قبل التصدير. يمكنك تنزيل النسخة المصححة.")
            edited_text = st.text_area(
                "OCR result (editable)",
                height=300,
                key="edited_text",
                label_visibility="collapsed",
            )
            if st.button("Generate downloads from corrected text", use_container_width=True):
                corrected = edited_text.strip()
                if corrected:
                    _generate_downloads(corrected)
                    st.success("Downloads updated using the corrected text.")
                else:
                    st.warning("Please keep at least some text before exporting.")
            download_col1, download_col2 = st.columns(2)
            with download_col1:
                if st.session_state["txt_bytes"]:
                    st.download_button("Download TXT", st.session_state["txt_bytes"], "arabic_ocr.txt", "text/plain", use_container_width=True)
            with download_col2:
                if st.session_state["docx_bytes"]:
                    st.download_button("Download DOCX", st.session_state["docx_bytes"], "arabic_ocr.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
        elif st.session_state["image_hash"]:
            st.info("No readable OCR text was returned for this image.")

    with right:
        st.markdown("## 🤖 AI Document Assistant")
        st.caption("Ask questions about your processed document.")
        example_question = None
        with st.container(border=True):
            if not st.session_state["ocr_text"]:
                st.markdown(
                    """
                    <div class="assistant-empty-state">
                        <div class="assistant-empty-avatar">🤖</div>
                        <div>Process a document to start chatting.</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif not st.session_state["chat_history"]:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write("مرحبًا! 👋")
                    st.write("يمكنني مساعدتك في فهم هذه الوثيقة والإجابة عن أسئلتك.")
                    st.write("جرّب تسأل:")
                    example_columns = st.columns(3)
                    examples = (
                        "ما موضوع هذه الوثيقة؟",
                        "ما أهم النقاط المذكورة في الوثيقة؟",
                        "هل يمكنك تلخيص هذه الرسالة؟",
                    )
                    for column, example in zip(example_columns, examples):
                        with column:
                            if st.button(
                                example,
                                key=f"example_question_{example}",
                                use_container_width=True,
                            ):
                                example_question = example

            for item in st.session_state["chat_history"]:
                with st.chat_message("user"):
                    st.write(item["question"])
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(item["answer"])
                    sources = item.get("sources", [])
                    if sources:
                        with st.expander("Sources / citations"):
                            for source in sources:
                                st.write(
                                    f"Source: {source.get('source') or 'Unknown'} · "
                                    f"Page: {source.get('page_id') or 'Unknown'} · "
                                    f"Chunk: {source.get('chunk_id') or 'Unknown'}"
                                )
            question = st.chat_input(
                "اسأل عن محتوى الوثيقة...",
                disabled=not bool(st.session_state["ocr_text"]),
            )
        with st.expander("OCR preview", expanded=False):
            st.text(st.session_state["ocr_text"] or "Process an image to view its OCR text.")
        submitted_question = example_question or question
        if submitted_question:
            with st.spinner("Thinking..."):
                _ask_question(submitted_question.strip())
            st.rerun()


if __name__ == "__main__":
    build_app()
