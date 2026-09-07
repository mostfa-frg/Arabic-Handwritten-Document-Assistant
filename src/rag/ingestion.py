from typing import Iterable

from src.rag.preprocessing import clean_arabic_text
from src.rag.chunking import create_chunks
from src.rag.embeddings import ArabicEmbeddingModel
from src.rag.vectorstore import ChromaVectorStore


def index_document(
    text: str,
    source: str,
    page_id: str,
    embedding_model: ArabicEmbeddingModel,
    vector_store: ChromaVectorStore,
) -> int:
    """
    Clean, chunk, embed, and index one document.
    """

    cleaned_text = clean_arabic_text(
        text
    )

    if not cleaned_text:
        return 0

    chunks = create_chunks(
        cleaned_text
    )

    if not chunks:
        return 0

    embeddings = embedding_model.encode(
        chunks
    )

    ids = [
        f"{page_id}_chunk_{i}"
        for i in range(len(chunks))
    ]

    metadatas = [
        {
            "source": source,
            "page_id": page_id,
            "chunk_id": i,
        }
        for i in range(len(chunks))
    ]

    vector_store.add_documents(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    return len(chunks)