from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = (
    "Omartificial-Intelligence-Space/"
    "Arabic-Triplet-Matryoshka-V2"
)


class ArabicEmbeddingModel:
    """
    Arabic sentence/document embedding model.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self.model = SentenceTransformer(
            model_name
        )

    def encode(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()