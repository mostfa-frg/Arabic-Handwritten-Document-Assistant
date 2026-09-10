"""Optional, non-destructive post-OCR spelling suggestions."""

from __future__ import annotations

from collections import Counter

from rapidfuzz import fuzz, process

from src.rag.hybrid import TOKEN_PATTERN, _tokens


# Change 4: provide pure, opt-in correction helpers that preserve the raw OCR
# text and its exact non-token structure for human review.
def build_vocabulary(lexical_index) -> Counter:
    """Build a token frequency table from an ``ArabicLexicalIndex``."""
    return Counter(lexical_index.document_frequency)


def correct_text(text: str, vocabulary: Counter, min_score: int = 85) -> str:
    """Return text with high-confidence unknown tokens replaced.

    Non-token characters, including whitespace and punctuation, are preserved
    exactly; callers can therefore present the result as a review suggestion
    rather than altering the original OCR text.
    """
    if not text or not vocabulary:
        return text

    parts = []
    cursor = 0
    vocabulary_keys = vocabulary.keys()
    for match in TOKEN_PATTERN.finditer(text):
        parts.append(text[cursor:match.start()])
        original = match.group(0)
        folded = original.casefold()
        if folded in vocabulary:
            replacement = original
        else:
            candidate = process.extractOne(
                folded,
                vocabulary_keys,
                scorer=fuzz.ratio,
                score_cutoff=min_score,
            )
            replacement = candidate[0] if candidate is not None else original
        parts.append(replacement)
        cursor = match.end()
    parts.append(text[cursor:])
    return "".join(parts)
