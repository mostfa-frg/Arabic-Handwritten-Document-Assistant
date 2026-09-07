from __future__ import annotations

import logging

from io import BytesIO
from pathlib import Path
from threading import Lock

from PIL import Image, UnidentifiedImageError

from src.cv.line_detector import LineDetector
from src.ocr.muharaf_ocr import MuharafOCR
from src.rag import chain
from src.rag.embeddings import ArabicEmbeddingModel
from src.rag.hybrid import ArabicLexicalIndex
from src.rag.retriever import ArabicRetriever
from src.rag.vectorstore import ChromaVectorStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEGMENTATION_MODEL = PROJECT_ROOT / "models" / "muharaf_seg_best.mlmodel"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
logger = logging.getLogger(__name__)


class RAGService:
    """Lazy application services shared by API requests."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._retriever = None
        self._lexical_index = None
        self._detector = None
        self._ocr = None

    def _load_rag(self) -> None:
        if self._retriever is not None:
            return
        with self._lock:
            if self._retriever is not None:
                return
            embedding_model = ArabicEmbeddingModel()
            vector_store = ChromaVectorStore()
            self._retriever = ArabicRetriever(embedding_model, vector_store)
            self._lexical_index = ArabicLexicalIndex.from_vector_store(vector_store)

    def _load_ocr(self) -> None:
        if self._detector is not None and self._ocr is not None:
            return
        with self._lock:
            if self._detector is not None and self._ocr is not None:
                return
            if not SEGMENTATION_MODEL.exists():
                raise FileNotFoundError(
                    f"Kraken segmentation model not found: {SEGMENTATION_MODEL}"
                )
            self._detector = LineDetector(
                backend="kraken",
                model_path=str(SEGMENTATION_MODEL),
                device="cpu",
            )
            self._ocr = MuharafOCR()

    def answer_text(self, question: str) -> dict:
        self._load_rag()
        return chain.answer_question(
            question=question,
            retriever=self._retriever,
            strategy="hybrid",
            lexical_index=self._lexical_index,
        )

    def image_to_text(self, payload: bytes) -> str:
        if not payload:
            raise ValueError("The uploaded image is empty.")
        if len(payload) > MAX_IMAGE_BYTES:
            raise ValueError("The uploaded image exceeds the 20 MB limit.")
        try:
            with Image.open(BytesIO(payload)) as image:
                image.verify()
            with Image.open(BytesIO(payload)) as image:
                page_image = image.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("The uploaded file is not a valid image.") from exc

        try:
            self._load_ocr()
        except Exception:
            logger.exception("OCR initialization failed")
            raise

        try:
            lines = self._detector.detect(page_image, return_crops=True)
        except Exception:
            logger.exception("Kraken line segmentation failed")
            raise

        try:
            return self._ocr.recognize_page(lines)
        except Exception:
            logger.exception(
                "Muharaf OCR failed after Kraken detected %d lines", len(lines)
            )
            raise

    def ocr_image(self, payload: bytes) -> str:
        return self.image_to_text(payload)

    def answer_image(self, question: str, payload: bytes) -> tuple[str, dict]:
        ocr_text = self.image_to_text(payload)
        self._load_rag()
        result = chain.answer_question(
            question=question,
            retrieval_query=f"{question}\n{ocr_text}".strip(),
            extra_context=ocr_text,
            retriever=self._retriever,
            strategy="hybrid",
            lexical_index=self._lexical_index,
        )
        return ocr_text, result


service = RAGService()
