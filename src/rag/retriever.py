from .embeddings import ArabicEmbeddingModel
from .vectorstore import ChromaVectorStore


class ArabicRetriever:

    def __init__(
        self,
        embedding_model: ArabicEmbeddingModel,
        vector_store: ChromaVectorStore,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ):
        query_embedding = (
            self.embedding_model.encode(
                [query]
            )[0]
        )

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )