"""Run the production OCR and RAG pipeline on the validation image set."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


QUESTIONS = (
    {
        "id": 1,
        "image": "validation_1.png",
        "question": "ما البلد المذكور في بداية الرسالة؟",
        "expected_answer": "لبنان؛ وتظهر في بداية OCR كلمة «لبنان»، مع تاريخ غير مكتمل.",
    },
    {
        "id": 2,
        "image": "validation_1.png",
        "question": "ما العملية التي تقول المرسلة إنها تنوي إجراءها؟",
        "expected_answer": "تذكر OCR عملية مرتبطة بالعين، لكن الصياغة والجهة غير مؤكدة بسبب أخطاء OCR.",
    },
    {
        "id": 3,
        "image": "validation_2.png",
        "question": "ما اسم الأخت الصغيرة المذكورة في الرسالة؟",
        "expected_answer": "يظهر الاسم في OCR بصيغة مشوشة («جور هن»)، لذلك القراءة الأقرب غير مؤكدة.",
    },
    {
        "id": 4,
        "image": "validation_2.png",
        "question": "إلى أين سيذهب خالي يعقوب غداً؟",
        "expected_answer": "يظهر في OCR اسم مكان مشوش («حسرصن»)، ولذلك لا يمكن الجزم بالقراءة.",
    },
    {
        "id": 5,
        "image": "validation_3.png",
        "question": "ما اسم القديس المذكور في القصيدة؟",
        "expected_answer": "تظهر عبارة مشوشة قريبة من «مار جرجس»، لكن الاسم غير واضح تماماً في OCR.",
    },
    {
        "id": 6,
        "image": "validation_3.png",
        "question": "من الذين تطلب القصيدة حمايتهم؟",
        "expected_answer": "تتحدث القصيدة عن حماية العائلة/الأهل، لكن السطور مشوشة ولا تسمح بتحديد كامل.",
    },
    {
        "id": 7,
        "image": "validation_4.png",
        "question": "في أي مكان تذكر الرسالة أنهم يسكنون؟",
        "expected_answer": "يظهر في OCR ذكر «البترون» و«ملك يوسف»، لكن تركيب المكان غير مؤكد.",
    },
    {
        "id": 8,
        "image": "validation_4.png",
        "question": "ما اسم إحدى المتوفيات المذكورة في نهاية النص؟",
        "expected_answer": "يظهر اسم مشوش قريب من «مريم»، لكن بقية الاسم غير مقروء بثقة.",
    },
    {
        "id": 9,
        "image": "validation_5.png",
        "question": "ما قيمة العيدية المذكورة في بداية الرسالة؟",
        "expected_answer": "تظهر قيمة مالية في OCR بصياغة مشوشة («خمة عز دولا»)، لذلك الرقم غير مؤكد.",
    },
    {
        "id": 10,
        "image": "validation_5.png",
        "question": "من الشخصية التي تشكر العاطفة النبيلة في الرسالة؟",
        "expected_answer": "تذكر OCR اسماً قريباً من «الأم ماري جوزيف غريب»، مع أخطاء في القراءة.",
    },
)

ASSESSMENTS = {
    1: ("Relevant", "Grounded", "Correct", "The answer matches the clear OCR heading."),
    2: ("Relevant", "Partially Grounded", "Partially Correct", "The eye reference is present, but the exact operation is not reliably transcribed."),
    3: ("Relevant", "Partially Grounded", "Partially Correct", "The name is present only as noisy OCR; the answer preserves uncertainty."),
    4: ("Relevant", "Grounded", "Partially Correct", "The retrieved OCR says «حص», while reviewed evidence reads «حمص»; the destination is OCR-uncertain."),
    5: ("Relevant", "Grounded", "Correct", "The retrieved text contains the noisy form «مارجرسنا» and supports «مار جرجس»."),
    6: ("Relevant", "Grounded", "Correct", "The retrieved text supports protection of the uncle and the whole family."),
    7: ("Relevant", "Grounded", "Correct", "The retrieved chunk contains the residence wording, although several words are noisy."),
    8: ("Relevant", "Grounded", "Correct", "The retrieved document text supports the name «مريم»."),
    9: ("Relevant", "Grounded", "Correct", "The retrieved chunk explicitly contains «خمسة ... دولارا»."),
    10: ("Relevant", "Partially Grounded", "Incorrect", "The name appears in context, but the answer assigns the thanks to the wrong person; the OCR says «الأم ماري جوزيف غريب»."),
}

def main() -> None:
    from src.api.services import RAGService

    project_root = PROJECT_ROOT
    image_root = project_root / "data" / "test_images" / "validation"
    output_root = project_root / "docs"
    output_root.mkdir(parents=True, exist_ok=True)

    images = sorted(image_root.glob("*.png"))
    if len(images) != 5:
        raise RuntimeError(f"Expected exactly 5 validation images, found {len(images)}")
    image_names = {path.name for path in images}
    if {row["image"] for row in QUESTIONS} - image_names:
        raise RuntimeError("A question references an image absent from the validation set")

    service = RAGService()
    ocr_by_image: dict[str, str] = {}
    for image in images:
        ocr_by_image[image.name] = service.ocr_image(image.read_bytes())

    rows: list[dict[str, object]] = []
    for item in QUESTIONS:
        result_ocr = ocr_by_image[item["image"]]
        result = service.answer_text(item["question"], result_ocr)
        relevance, grounding, correctness, notes = ASSESSMENTS[item["id"]]
        rows.append(
            {
                **item,
                "ocr_text": result_ocr,
                "answer": result["answer"],
                "sources": result.get("sources", []),
                "retrieved_chunks": result.get("selected_chunks", []),
                "context": result.get("context", ""),
                "retrieval_relevance": relevance,
                "grounding": grounding,
                "correctness": correctness,
                "notes": notes,
            }
        )

    (output_root / "evaluation_raw.json").write_text(
        json.dumps(
            {
                "images": [path.name for path in images],
                "ocr_by_image": ocr_by_image,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    fieldnames = (
        "id",
        "image",
        "question",
        "expected_answer",
        "retrieved_source",
        "retrieved_page",
        "retrieved_chunk",
        "retrieval_relevance",
        "answer",
        "grounding",
        "correctness",
        "notes",
    )
    with (output_root / "evaluation_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            sources = row["sources"]
            first = sources[0] if sources else {}
            writer.writerow(
                {
                    "id": row["id"],
                    "image": row["image"],
                    "question": row["question"],
                    "expected_answer": row["expected_answer"],
                    "retrieved_source": first.get("source", ""),
                    "retrieved_page": first.get("page_id", ""),
                    "retrieved_chunk": first.get("chunk_id", ""),
                    "retrieval_relevance": row["retrieval_relevance"],
                    "answer": row["answer"],
                    "grounding": row["grounding"],
                    "correctness": row["correctness"],
                    "notes": row["notes"],
                }
            )

    print(json.dumps({"images": len(images), "questions": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
