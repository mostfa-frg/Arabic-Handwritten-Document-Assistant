from pathlib import Path

import chromadb


class ChromaVectorStore:

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "handwritten_documents",
    ):
        Path(
            persist_directory
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = (
            self.client.get_or_create_collection(
                name=collection_name
            )
        )

    def add_documents(
        self,
        documents: list[str],
        embeddings: list[list[float]],
        ids: list[str],
        metadatas: list[dict],
    ) -> None:

        self.collection.upsert(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    def count(self) -> int:
        return self.collection.count()