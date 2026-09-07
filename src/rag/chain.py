from __future__ import annotations

import sys

from openai import OpenAI

from src import config
from src.rag.hybrid import ArabicLexicalIndex, reciprocal_rank_fusion


SYSTEM_PROMPT = """أنت مساعد لفهم الوثائق العربية المكتوبة بخط اليد.
أجب عن سؤال المستخدم اعتمادًا على السياق المسترجع فقط.
إذا لم يحتوي السياق على معلومات كافية للإجابة، فقل بوضوح:
«لا توجد معلومات كافية في الوثائق المسترجعة للإجابة عن هذا السؤال».
لا تخترع معلومات ولا تستخدم معرفة خارج السياق.
ركّز على الوثيقة أو الصفحة التي تدعمها المقاطع المختارة، ولا تخلط بين وثائق
مختلفة إلا إذا كان السؤال يتطلب ذلك. إذا كان النص المشوش بسبب OCR لا يسمح
بالجزم، فاذكر عدم اليقين بدل اختراع أسماء أو تواريخ أو أحداث.
أجب باللغة العربية وباختصار، واذكر المصادر ذات الصلة عندما تكون متاحة."""


def _result_to_chunks(result: dict) -> list[dict]:
    documents = result.get("documents", [[]])[0] or []
    metadatas = result.get("metadatas", [[]])[0] or []
    distances = result.get("distances", [[]])[0] or []

    chunks = []
    for index, document in enumerate(documents):
        if not document or not any(character.isalnum() for character in str(document)):
            continue
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None
        chunks.append(
            {
                "text": document,
                "source": metadata.get("source"),
                "page_id": metadata.get("page_id"),
                "chunk_id": metadata.get("chunk_id"),
                "distance": distance,
            }
        )
    return chunks


def _group_key(chunk: dict) -> tuple | None:
    source = chunk.get("source")
    page_id = chunk.get("page_id")
    if source is None and page_id is None:
        return None
    return source, page_id


def _chunk_sort_key(chunk: dict) -> tuple:
    chunk_id = chunk.get("chunk_id")
    try:
        return 0, int(chunk_id)
    except (TypeError, ValueError):
        return 1, str(chunk_id)


def _select_context_chunks(chunks: list[dict], top_k: int) -> list[dict]:
    """Prefer a well-supported page while safely falling back to semantic top-k."""
    if not chunks:
        return []

    groups: dict[tuple, list[dict]] = {}
    for chunk in chunks:
        key = _group_key(chunk)
        if key is not None:
            groups.setdefault(key, []).append(chunk)

    if not groups:
        return chunks[:top_k]

    # A page supported by multiple retrieved chunks is stronger evidence than
    # an isolated hit. Missing/incomplete metadata remains on the fallback path.
    supported_groups = [
        group for group in groups.values() if len(group) >= 2
    ]
    if not supported_groups:
        return chunks[:top_k]

    def group_score(group: list[dict]) -> tuple:
        distances = [
            chunk["distance"]
            for chunk in group
            if isinstance(chunk.get("distance"), (int, float))
        ]
        best_distance = min(distances) if distances else float("inf")
        return best_distance, -len(group)

    selected_group = min(supported_groups, key=group_score)
    return sorted(selected_group, key=_chunk_sort_key)[:top_k]


def _select_context_chunks_semantic_page_aware(
    chunks: list[dict], top_k: int
) -> list[dict]:
    """Select one page using strongest semantic evidence before chunk count."""
    if not chunks:
        return []

    groups: dict[tuple, list[dict]] = {}
    for chunk in chunks:
        key = _group_key(chunk)
        if key is not None:
            groups.setdefault(key, []).append(chunk)
    if not groups:
        return chunks[:top_k]

    def page_score(group: list[dict]) -> tuple[float, float]:
        distances = sorted(
            float(chunk["distance"])
            for chunk in group
            if isinstance(chunk.get("distance"), (int, float))
        )
        if not distances:
            return float("-inf"), 0.0
        # The best semantic hit is the primary signal. The next two hits only
        # break close ties, so many weak chunks cannot beat one strong hit.
        strengths = [1.0 / (1.0 + distance) for distance in distances[:3]]
        weights = (0.75, 0.17, 0.08)
        secondary = sum(
            weight * strength for weight, strength in zip(weights[1:], strengths[1:])
        )
        return -distances[0], secondary

    selected_group = max(groups.values(), key=page_score)
    return sorted(selected_group, key=_chunk_sort_key)[:top_k]


def _sources_from_chunks(chunks: list[dict]) -> list[dict]:
    sources = []
    for chunk in chunks:
        source = {
            "source": chunk.get("source"),
            "page_id": chunk.get("page_id"),
            "chunk_id": chunk.get("chunk_id"),
        }
        if source not in sources:
            sources.append(source)
    return sources


def _format_context(chunks: list[dict]) -> str:
    context_parts = []
    for index, chunk in enumerate(chunks):
        source = {
            "source": chunk.get("source"),
            "page_id": chunk.get("page_id"),
            "chunk_id": chunk.get("chunk_id"),
        }
        context_parts.append(
            f"[المقطع {index + 1} | المصدر: {source['source']} | "
            f"الصفحة: {source['page_id']} | المقطع: {source['chunk_id']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(context_parts)


def retrieve_context(
    question: str,
    retriever,
    top_k: int,
    strategy: str = "current",
    lexical_index: ArabicLexicalIndex | None = None,
) -> dict:
    """Return raw semantic hits and the metadata-aware context selection."""
    candidate_k = max(top_k * 3, top_k)
    raw_result = retriever.retrieve(question, top_k=candidate_k)
    retrieved_chunks = _result_to_chunks(raw_result)
    if strategy == "hybrid":
        if lexical_index is None:
            lexical_index = ArabicLexicalIndex.from_vector_store(
                retriever.vector_store
            )
        lexical_chunks = lexical_index.search(question, top_k=candidate_k)
        selected_chunks = reciprocal_rank_fusion(
            retrieved_chunks,
            lexical_chunks,
            top_k=top_k,
        )
        retrieved_chunks = selected_chunks
    elif strategy == "semantic_page_aware":
        selected_chunks = _select_context_chunks_semantic_page_aware(
            retrieved_chunks, top_k
        )
    elif strategy == "current":
        selected_chunks = _select_context_chunks(retrieved_chunks, top_k)
    elif strategy == "semantic":
        selected_chunks = retrieved_chunks[:top_k]
    else:
        raise ValueError(
            "strategy must be one of: semantic, current, semantic_page_aware, hybrid"
        )
    return {
        "retrieved_chunks": retrieved_chunks,
        "selected_chunks": selected_chunks,
        "strategy": strategy,
    }


def _console_print(text: str) -> None:
    """Print Arabic safely when a Windows console uses a legacy code page."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        print(text.encode(encoding, errors="replace").decode(encoding))


def debug_retrieval(query: str, retriever, top_k: int = config.TOP_K) -> dict:
    """Print raw and selected chunks for retrieval inspection."""
    retrieval = retrieve_context(query, retriever, top_k)
    for label, chunks in (
        ("Semantic candidates", retrieval["retrieved_chunks"]),
        ("Selected context", retrieval["selected_chunks"]),
    ):
        _console_print(f"\n{label} ({len(chunks)}):")
        for rank, chunk in enumerate(chunks, start=1):
            preview = " ".join(str(chunk["text"]).split())[:240]
            _console_print(
                f"{rank}. distance={chunk['distance']} | "
                f"source={chunk['source']} | page_id={chunk['page_id']} | "
                f"chunk_id={chunk['chunk_id']}\n   {preview}"
            )
    return retrieval


def answer_question(
    question: str,
    retriever,
    top_k: int | None = None,
    strategy: str = "current",
    lexical_index: ArabicLexicalIndex | None = None,
    retrieval_query: str | None = None,
    extra_context: str | None = None,
) -> dict:
    """Retrieve Arabic document context and answer through the Groq API."""
    api_key = config.GROQ_API_KEY
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Set it as an environment variable before "
            "calling answer_question()."
        )
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    top_k = config.TOP_K if top_k is None else top_k
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    retrieval = retrieve_context(
        retrieval_query or question,
        retriever,
        top_k,
        strategy=strategy,
        lexical_index=lexical_index,
    )
    retrieved_chunks = retrieval["retrieved_chunks"]
    selected_chunks = retrieval["selected_chunks"]
    context = _format_context(selected_chunks)
    if extra_context and extra_context.strip():
        context = (
            f"{context}\n\n[النص المستخرج من الصورة]\n{extra_context.strip()}"
            if context
            else extra_context.strip()
        )
    sources = _sources_from_chunks(selected_chunks)
    if not context:
        context = "لا يوجد سياق مسترجع."

    client = OpenAI(api_key=api_key, base_url=config.GROQ_BASE_URL)
    completion = client.chat.completions.create(
        model=config.GROQ_MODEL,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"السياق:\n{context}\n\nالسؤال:\n{question.strip()}",
            },
        ],
    )
    answer = completion.choices[0].message.content
    if not answer:
        raise RuntimeError("Groq returned an empty answer.")

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": selected_chunks,
        "retrieved_candidates": retrieved_chunks,
        "selected_chunks": selected_chunks,
        "context": context,
    }
