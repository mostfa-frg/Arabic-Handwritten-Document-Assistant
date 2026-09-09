from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import APIError
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse

from src.api.documents import write_docx_file, write_text_file
from src.api.services import RAGService, service

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    ocr_text: str | None = None


class DocumentRequest(BaseModel):
    text: str = Field(..., min_length=1)


class Source(BaseModel):
    source: str | None = None
    page_id: str | None = None
    chunk_id: int | str | None = None


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


class ImageQueryResponse(QueryResponse):
    ocr_text: str


class OCRResponse(BaseModel):
    ocr_text: str


def get_service() -> RAGService:
    return service


app = FastAPI(
    title="Arabic Handwritten Document RAG API",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
router = APIRouter()


def _handle_service_error(
    exc: ValueError
    | FileNotFoundError
    | RuntimeError
    | ImportError
    | OSError
    | APIError,
) -> None:
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.exception("API service failure: %s: %s", type(exc).__name__, exc)
    raise HTTPException(
        status_code=503,
        detail=f"{type(exc).__name__}: {exc}",
    ) from exc


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/ocr", response_model=OCRResponse)
async def ocr(
    image: UploadFile = File(...),
    rag: RAGService = Depends(get_service),
):
    try:
        ocr_text = rag.ocr_image(await image.read())
    except (ValueError, FileNotFoundError, RuntimeError, ImportError, OSError) as exc:
        _handle_service_error(exc)
    return OCRResponse(ocr_text=ocr_text)


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, rag: RAGService = Depends(get_service)):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")
    try:
        result = rag.answer_text(question, request.ocr_text)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError, OSError, APIError) as exc:
        _handle_service_error(exc)
    return QueryResponse(
        question=question,
        answer=result["answer"],
        sources=result.get("sources", []),
    )


def _document_response(text: str, extension: str, writer, background_tasks: BackgroundTasks):
    temp_file = tempfile.NamedTemporaryFile(
        prefix="arabic_document_", suffix=extension, delete=False
    )
    temp_path = Path(temp_file.name)
    temp_file.close()
    writer(text, temp_path)
    background_tasks.add_task(os.unlink, temp_path)
    return FileResponse(
        temp_path,
        media_type=(
            "text/plain; charset=utf-8"
            if extension == ".txt"
            else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ),
        filename=f"arabic_ocr{extension}",
        background=background_tasks,
    )


@router.post("/documents/txt")
def document_txt(
    request: DocumentRequest, background_tasks: BackgroundTasks
):
    return _document_response(
        request.text, ".txt", write_text_file, background_tasks
    )


@router.post("/documents/docx")
def document_docx(
    request: DocumentRequest, background_tasks: BackgroundTasks
):
    return _document_response(
        request.text, ".docx", write_docx_file, background_tasks
    )


@router.post("/query-image", response_model=ImageQueryResponse)
async def query_image(
    question: str = Form(...),
    image: UploadFile = File(...),
    rag: RAGService = Depends(get_service),
):
    question = question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")
    try:
        payload = await image.read()
        ocr_text, result = rag.answer_image(question, payload)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError, OSError, APIError) as exc:
        _handle_service_error(exc)
    return ImageQueryResponse(
        question=question,
        ocr_text=ocr_text,
        answer=result["answer"],
        sources=result.get("sources", []),
    )


app.include_router(router)
