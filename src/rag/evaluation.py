from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.rag.chain import _format_context, _sources_from_chunks, retrieve_context
from src.rag.hybrid import ArabicLexicalIndex, reciprocal_rank_fusion


@dataclass(frozen=True)
class EvaluationQuestion:
    question: str
    expected_page_id: str
    expected_answer_contains: tuple[str, ...] = ()


# These page IDs are present in the current persisted collection. The answer
# evidence is intentionally limited to facts visible in the indexed OCR/page
# metadata; questions without reliable answer evidence remain retrieval-only.
EVALUATION_QUESTIONS = (
    EvaluationQuestion(
        "ما اسم الشخص المذكور في عنوان هذه الرسالة؟",
        "2015 5-03 El-Khouri_Letter to Jennie Jabaley from Lebanon Oct18 1960_1",
        ("جيني",),
    ),
    EvaluationQuestion(
        "ما السنة المذكورة في معرّف هذه الوثيقة؟",
        "2015 5-07 El-Khouri_Letter to Joseph from Lebanon Dec17 1959_1",
        ("1959",),
    ),
    EvaluationQuestion(
        "ما السنة المذكورة في معرّف هذه الوثيقة؟",
        "2015 5-07b El-Khouri_Letter to Joseph from Lebanon Dec17 1959_2_1",
        ("1959",),
    ),
    EvaluationQuestion(
        "ما السنة المذكورة في معرّف هذه الوثيقة؟",
        "2015 5-08b El-Khouri_Letter to Joseph from Lebanon Dec19 1957_2_1",
        ("1957",),
    ),
    EvaluationQuestion(
        "ما السنة المذكورة في معرّف هذه الوثيقة؟",
        "2015 5-08c El-Khouri_Letter to Joseph from Lebanon Dec19 1957_3",
        ("1957",),
    ),
    EvaluationQuestion(
        "ما السنة المذكورة في معرّف هذه الوثيقة؟",
        "2015 5-09b El-Khouri_Letter to Joseph from Lebanon Dec21 1960_2",
        ("1960",),
    ),
    EvaluationQuestion(
        "ما السنة المذكورة في معرّف هذه الوثيقة؟",
        "2015 5-13 El-Khouri_Letter to Joseph from Lebanon Feb29 1960 1",
        ("1960",),
    ),
    EvaluationQuestion(
        "ما نوع المعاملة القانونية المذكورة في هذا النص؟",
        "AF_282r",
        ("مقايضة",),
    ),
    EvaluationQuestion(
        "ما الإجراء المالي المذكور في هذا النص؟",
        "AF_293r",
        ("الدفع",),
    ),
    EvaluationQuestion(
        "ما رقم الصفحة الظاهر في معرّف الوثيقة؟",
        "AF_303r",
        ("AF_303r",),
    ),
)


def build_verified_ocr_questions(
    retriever,
    count: int = 10,
) -> tuple[EvaluationQuestion, ...]:
    """Build deterministic questions whose answer term occurs in indexed OCR."""
    lexical_index = ArabicLexicalIndex.from_vector_store(retriever.vector_store)
    page_terms: dict[str, list[str]] = {}
    term_pages: dict[str, set[str]] = {}
    for document_index, metadata in enumerate(lexical_index.metadatas):
        page_id = str(metadata.get("page_id"))
        terms = sorted(
            {
                token
                for token in lexical_index.tokenized[document_index]
                if len(token) >= 5
            },
            key=lambda token: (-len(token), token),
        )
        page_terms.setdefault(page_id, []).extend(terms)
        for term in set(terms):
            term_pages.setdefault(term, set()).add(page_id)

    candidates = []
    for page_id, terms in page_terms.items():
        for term in terms:
            if len(term_pages.get(term, set())) == 1:
                candidates.append((page_id, term))
                break
    candidates.sort(key=lambda item: (item[0], item[1]))
    if len(candidates) < count:
        raise ValueError(
            f"Only {len(candidates)} pages have unique OCR terms; "
            f"cannot construct {count} verified questions."
        )
    return tuple(
        EvaluationQuestion(
            question=f"هل يذكر النص الكلمة «{term}»؟",
            expected_page_id=page_id,
            expected_answer_contains=(term,),
        )
        for page_id, term in candidates[:count]
    )


def _page_ids(chunks: Iterable[dict]) -> list[str]:
    return list(dict.fromkeys(
        str(chunk["page_id"])
        for chunk in chunks
        if chunk.get("page_id") is not None
    ))


def _answer_matches(answer: str, expected_terms: tuple[str, ...]) -> bool | None:
    if not expected_terms:
        return None
    normalized_answer = answer.casefold()
    return any(term.casefold() in normalized_answer for term in expected_terms)


def _groundedness(answer: str, context: str) -> bool:
    """Conservative lexical support check for noisy OCR evaluation."""
    if not answer.strip():
        return False
    insufficient = "لا توجد معلومات كافية" in answer
    if insufficient:
        return True
    answer_terms = {
        token for token in answer.split()
        if len(token.strip("،.!؟:؛()[]«»\"'")) >= 4
    }
    context_folded = context.casefold()
    return bool(answer_terms) and any(
        token.casefold() in context_folded for token in answer_terms
    )


def _validate_questions(
    questions: Iterable[EvaluationQuestion],
    retriever,
) -> None:
    collection = getattr(getattr(retriever, "vector_store", None), "collection", None)
    if collection is None:
        raise TypeError(
            "The retriever must expose vector_store.collection so evaluation "
            "can validate expected page IDs against the current index."
        )
    metadata = collection.get(include=["metadatas"]).get("metadatas", [])
    available = {str(item.get("page_id")) for item in metadata if item.get("page_id")}
    missing = sorted({q.expected_page_id for q in questions} - available)
    if missing:
        raise ValueError(
            "Evaluation questions reference page IDs absent from the current "
            f"index: {missing}"
        )


def audit_questions(
    retriever,
    questions: Iterable[EvaluationQuestion] = EVALUATION_QUESTIONS,
) -> pd.DataFrame:
    """Show expected-page chunks and whether answer terms occur in OCR text."""
    questions = tuple(questions)
    _validate_questions(questions, retriever)
    collection = retriever.vector_store.collection
    indexed = collection.get(include=["documents", "metadatas"])
    rows = []
    for item in questions:
        page_chunks = [
            {
                "chunk_id": metadata.get("chunk_id"),
                "text": document,
            }
            for document, metadata in zip(
                indexed["documents"], indexed["metadatas"]
            )
            if metadata.get("page_id") == item.expected_page_id
        ]
        joined_text = " ".join(chunk["text"] for chunk in page_chunks).casefold()
        term_hits = [
            term for term in item.expected_answer_contains
            if term.casefold() in joined_text
        ]
        rows.append(
            {
                "question": item.question,
                "expected_page_id": item.expected_page_id,
                "indexed_chunks": page_chunks,
                "page_exists": bool(page_chunks),
                "expected_terms": item.expected_answer_contains,
                "terms_found_in_ocr": tuple(term_hits),
                "ground_truth_status": (
                    "ocr_evidence"
                    if term_hits
                    else "metadata_only_or_ambiguous"
                ),
            }
        )
    return pd.DataFrame(rows)


def evaluate_retrieval(
    retriever,
    questions: Iterable[EvaluationQuestion] = EVALUATION_QUESTIONS,
    top_k: int = 5,
    run_generation: bool = True,
    strategy: str = "current",
    lexical_index: ArabicLexicalIndex | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run reproducible retrieval/generation evaluation on the current index."""
    questions = tuple(questions)
    _validate_questions(questions, retriever)

    answer_function = None
    if run_generation:
        from src.rag.chain import answer_question

        answer_function = answer_question

    rows: list[dict[str, Any]] = []
    for item in questions:
        if strategy == "hybrid":
            if lexical_index is None:
                lexical_index = ArabicLexicalIndex.from_vector_store(
                    retriever.vector_store
                )
            semantic = retrieve_context(
                item.question, retriever, top_k, strategy="semantic"
            )["selected_chunks"]
            lexical = lexical_index.search(item.question, top_k=top_k * 3)
            candidates = reciprocal_rank_fusion(
                semantic, lexical, top_k=top_k * 3
            )
            selected = candidates[:top_k]
            retrieval = {
                "retrieved_chunks": candidates,
                "selected_chunks": selected,
            }
        else:
            retrieval = retrieve_context(
                item.question, retriever, top_k, strategy=strategy
            )
        candidates = retrieval["retrieved_chunks"]
        selected = retrieval["selected_chunks"]
        semantic_top_k = candidates[:top_k]
        context = _format_context(selected)

        answer = ""
        sources = _sources_from_chunks(selected)
        if answer_function is not None:
            result = answer_function(item.question, retriever, top_k=top_k)
            answer = result["answer"]
            sources = result["sources"]
            context = result["context"]

        candidate_pages = _page_ids(candidates)
        selected_pages = _page_ids(selected)
        semantic_pages = _page_ids(semantic_top_k)
        rows.append(
            {
                "question": item.question,
                "expected_page_id": item.expected_page_id,
                "retrieved_candidates": candidates,
                "selected_chunks": selected,
                "selected_page_ids": selected_pages,
                "semantic_page_ids": semantic_pages,
                "expected_in_candidates": item.expected_page_id in candidate_pages,
                "expected_in_semantic_top_k": item.expected_page_id in semantic_pages,
                "expected_in_selected": item.expected_page_id in selected_pages,
                "answer": answer,
                "sources": sources,
                "context": context,
                "answer_correct": (
                    _answer_matches(answer, item.expected_answer_contains)
                    if answer
                    else None
                ),
                "grounded": _groundedness(answer, context) if answer else None,
                "strategy": strategy,
            }
        )

    results = pd.DataFrame(rows)
    metrics = {
        "semantic_retrieval_hit_at_k": float(
            results["expected_in_semantic_top_k"].mean()
        ),
        "document_aware_retrieval_hit_at_k": float(
            results["expected_in_selected"].mean()
        ),
        "semantic_selected_page_accuracy": float(
            results.apply(
                lambda row: row["expected_page_id"] in row["semantic_page_ids"],
                axis=1,
            ).mean()
        ),
        "document_aware_selected_page_accuracy": float(
            results["expected_in_selected"].mean()
        ),
    }
    answer_rows = results["answer_correct"].dropna()
    if not answer_rows.empty:
        metrics["answer_correctness"] = float(answer_rows.mean())
    grounded_rows = results["grounded"].dropna()
    if not grounded_rows.empty:
        metrics["groundedness"] = float(grounded_rows.mean())
    return results, metrics


def comparison_table(results: pd.DataFrame) -> pd.DataFrame:
    """Return a compact ordinary-vs-document-aware comparison table."""
    return pd.DataFrame(
        [
            {
                "strategy": "ordinary semantic top-k",
                "retrieval_hit_at_k": results["expected_in_semantic_top_k"].mean(),
                "selected_page_accuracy": results.apply(
                    lambda row: row["expected_page_id"] in row["semantic_page_ids"],
                    axis=1,
                ).mean(),
            },
            {
                "strategy": "document-aware selected context",
                "retrieval_hit_at_k": results["expected_in_selected"].mean(),
                "selected_page_accuracy": results["expected_in_selected"].mean(),
            },
        ]
    )


def compare_strategies(
    retriever,
    questions: Iterable[EvaluationQuestion] = EVALUATION_QUESTIONS,
    top_k: int = 5,
    lexical_index: ArabicLexicalIndex | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run semantic, current, and improved retrieval without generation."""
    frames = []
    metrics = []
    for strategy in (
        "semantic",
        "current",
        "semantic_page_aware",
        "hybrid",
    ):
        result, metric = evaluate_retrieval(
            retriever,
            questions=questions,
            top_k=top_k,
            run_generation=False,
            strategy=strategy,
            lexical_index=lexical_index,
        )
        frames.append(result)
        metrics.append(
            {
                "strategy": strategy,
                "hit_at_k": metric[
                    "semantic_retrieval_hit_at_k"
                    if strategy == "semantic"
                    else "document_aware_retrieval_hit_at_k"
                ],
                "selected_page_accuracy": metric[
                    "semantic_selected_page_accuracy"
                    if strategy == "semantic"
                    else "document_aware_selected_page_accuracy"
                ],
            }
        )
    per_question = pd.concat(frames, ignore_index=True)
    return pd.DataFrame(metrics), per_question


def retrieval_audit(
    retriever,
    questions: Iterable[EvaluationQuestion] = EVALUATION_QUESTIONS,
    top_k: int = 5,
    strategy: str = "current",
) -> pd.DataFrame:
    """Return candidate-level details for diagnosing retrieval decisions."""
    lexical_index = None
    if strategy == "hybrid":
        lexical_index = ArabicLexicalIndex.from_vector_store(
            retriever.vector_store
        )
    rows = []
    for item in questions:
        if strategy == "hybrid":
            semantic = retrieve_context(
                item.question, retriever, top_k, strategy="semantic"
            )["selected_chunks"]
            lexical = lexical_index.search(item.question, top_k=top_k * 3)
            candidates = reciprocal_rank_fusion(
                semantic, lexical, top_k=top_k * 3
            )
            retrieval = {
                "retrieved_chunks": candidates,
                "selected_chunks": candidates[:top_k],
            }
        else:
            retrieval = retrieve_context(
                item.question, retriever, top_k, strategy=strategy
            )
        selected = retrieval["selected_chunks"]
        selected_pages = _page_ids(selected)
        for rank, chunk in enumerate(retrieval["retrieved_chunks"], start=1):
            rows.append(
                {
                    "question": item.question,
                    "expected_page_id": item.expected_page_id,
                    "rank": rank,
                    "distance": chunk.get("distance"),
                    "page_id": chunk.get("page_id"),
                    "chunk_id": chunk.get("chunk_id"),
                    "text_preview": " ".join(str(chunk["text"]).split())[:240],
                    "selected": chunk in selected,
                    "selected_page_ids": selected_pages,
                    "selected_page_reason": (
                        "selected by " + strategy
                        if chunk in selected
                        else "not selected"
                    ),
                }
            )
    return pd.DataFrame(rows)


def readable_results(results: pd.DataFrame) -> pd.DataFrame:
    """Select scalar fields suitable for notebook or terminal display."""
    return results[
        [
            "question",
            "expected_page_id",
            "semantic_page_ids",
            "selected_page_ids",
            "expected_in_semantic_top_k",
            "expected_in_candidates",
            "expected_in_selected",
            "answer_correct",
            "grounded",
            "answer",
            "sources",
        ]
    ]
