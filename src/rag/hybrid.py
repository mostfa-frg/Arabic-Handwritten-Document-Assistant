from __future__ import annotations

import math
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9\u0600-\u06ff]+")


def _tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).casefold())


class ArabicLexicalIndex:
    """Small read-only BM25 index over the existing Chroma documents."""

    def __init__(self, documents: list[str], metadatas: list[dict]):
        self.documents = documents
        self.metadatas = metadatas
        self.tokenized = [_tokens(document) for document in documents]
        self.term_frequencies = [Counter(tokens) for tokens in self.tokenized]
        self.document_frequency = Counter(
            token for tokens in self.tokenized for token in set(tokens)
        )
        self.average_length = (
            sum(len(tokens) for tokens in self.tokenized) / len(self.tokenized)
            if self.tokenized
            else 0.0
        )

    @classmethod
    def from_vector_store(cls, vector_store) -> "ArabicLexicalIndex":
        result = vector_store.collection.get(include=["documents", "metadatas"])
        return cls(result["documents"], result["metadatas"])

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_tokens = _tokens(query)
        total_documents = len(self.documents)
        if not query_tokens or not total_documents:
            return []

        k1, b = 1.5, 0.75
        scored = []
        for index, frequencies in enumerate(self.term_frequencies):
            length = len(self.tokenized[index])
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                document_frequency = self.document_frequency[token]
                inverse_frequency = math.log(
                    1.0
                    + (total_documents - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                normalization = 1.0 - b + b * length / max(self.average_length, 1.0)
                score += inverse_frequency * (
                    frequency * (k1 + 1.0)
                    / (frequency + k1 * normalization)
                )
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {
                "text": self.documents[index],
                "source": self.metadatas[index].get("source"),
                "page_id": self.metadatas[index].get("page_id"),
                "chunk_id": self.metadatas[index].get("chunk_id"),
                "lexical_score": score,
            }
            for score, index in scored[:top_k]
        ]


def reciprocal_rank_fusion(
    semantic_chunks: list[dict],
    lexical_chunks: list[dict],
    top_k: int,
    rank_constant: int = 60,
) -> list[dict]:
    """Combine ranked semantic and lexical results without concatenation."""
    by_id = {}
    scores = Counter()
    for rank, chunk in enumerate(semantic_chunks, start=1):
        key = (chunk.get("source"), chunk.get("page_id"), chunk.get("chunk_id"))
        by_id[key] = chunk
        scores[key] += 1.0 / (rank_constant + rank)
    for rank, chunk in enumerate(lexical_chunks, start=1):
        key = (chunk.get("source"), chunk.get("page_id"), chunk.get("chunk_id"))
        by_id[key] = chunk
        scores[key] += 1.0 / (rank_constant + rank)

    ranked_keys = sorted(scores, key=lambda key: (-scores[key], str(key)))
    return [
        {**by_id[key], "hybrid_score": scores[key]}
        for key in ranked_keys[:top_k]
    ]
