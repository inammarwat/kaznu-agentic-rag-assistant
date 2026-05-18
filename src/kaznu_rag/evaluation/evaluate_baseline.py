import argparse
import json
import logging
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from kaznu_rag.rag.llm_client import initialize_llm

logger = logging.getLogger(__name__)


EVALUATION_SYSTEM_PROMPT = """
You are an expert evaluator of Retrieval-Augmented Generation (RAG) systems.

Evaluate the assistant answer using ONLY:
1. the user question,
2. the retrieved context,
3. the generated answer.

Do not use outside knowledge.

Return ONLY valid JSON with this schema:

{
  "faithfulness_score": 1-5,
  "answer_relevance_score": 1-5,
  "context_relevance_score": 1-5,
  "completeness_score": 1-5,
  "citation_quality_score": 1-5,
  "hallucination_score": 1-5,
  "hallucination_detected": true or false,
  "unsupported_claims": ["claim 1", "claim 2"],
  "short_reason": "brief explanation"
}

Scoring definitions:
- faithfulness_score: Whether the answer is supported by retrieved context.
- answer_relevance_score: Whether the answer directly answers the question.
- context_relevance_score: Whether retrieved context is relevant to the question.
- completeness_score: Whether the answer covers the important parts found in context.
- citation_quality_score: Whether citations/sources are present and useful.
- hallucination_score: 5 means no hallucination, 1 means severe hallucination.
- hallucination_detected: true if the answer contains unsupported or invented claims.

Important:
If the answer includes a fact not supported by the retrieved context, mark it as hallucination.
If the context does not contain enough information and the answer admits uncertainty, that is not hallucination.
""".strip()


EVALUATION_USER_PROMPT = """
Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate now. Return only JSON.
""".strip()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_context_for_evaluation(retrieved_context: List[Dict[str, Any]]) -> str:
    blocks = []

    for item in retrieved_context:
        rank = item.get("rank")
        score = item.get("score")
        text = item.get("text", "")
        metadata = item.get("metadata", {})

        source_name = metadata.get("source_name", "Unknown source")
        page_number = metadata.get("page_number", "")
        url = metadata.get("url", "")
        content_type = metadata.get("content_type", "")

        source_parts = [f"rank={rank}", f"score={score}", f"type={content_type}"]

        if source_name:
            source_parts.append(f"source={source_name}")

        if page_number not in {"", None}:
            source_parts.append(f"page={page_number}")

        if url:
            source_parts.append(f"url={url}")

        blocks.append(
            "[Retrieved Source: "
            + " | ".join(source_parts)
            + "]\n"
            + text
        )

    return "\n\n".join(blocks)


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def normalize_score(value: Any, default: int = 0) -> int:
    try:
        value = int(value)
    except Exception:
        return default

    if value < 1:
        return 1

    if value > 5:
        return 5

    return value


def normalize_evaluation(raw_eval: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "faithfulness_score": normalize_score(raw_eval.get("faithfulness_score")),
        "answer_relevance_score": normalize_score(raw_eval.get("answer_relevance_score")),
        "context_relevance_score": normalize_score(raw_eval.get("context_relevance_score")),
        "completeness_score": normalize_score(raw_eval.get("completeness_score")),
        "citation_quality_score": normalize_score(raw_eval.get("citation_quality_score")),
        "hallucination_score": normalize_score(raw_eval.get("hallucination_score")),
        "hallucination_detected": bool(raw_eval.get("hallucination_detected", False)),
        "unsupported_claims": raw_eval.get("unsupported_claims", []),
        "short_reason": str(raw_eval.get("short_reason", "")),
    }


def evaluate_one_record(
    record: Dict[str, Any],
    evaluator_llm,
) -> Dict[str, Any]:
    question = record.get("question", "")
    answer = record.get("answer", "")

    # Baseline RAG stores sources in "retrieved_context".
    # Agentic RAG stores final validated sources in "validated_context".
    retrieved_context = (
        record.get("retrieved_context")
        or record.get("validated_context")
        or []
    )

    if record.get("retrieved_context"):
        context_source_field = "retrieved_context"
    elif record.get("validated_context"):
        context_source_field = "validated_context"
    else:
        context_source_field = "missing_context"

    context_text = format_context_for_evaluation(retrieved_context)

    prompt = EVALUATION_USER_PROMPT.format(
        question=question,
        context=context_text,
        answer=answer,
    )

    response = evaluator_llm.invoke(
        [
            SystemMessage(content=EVALUATION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    parsed = extract_json_from_text(response.content)

    if parsed is None:
        parsed = {
            "faithfulness_score": 0,
            "answer_relevance_score": 0,
            "context_relevance_score": 0,
            "completeness_score": 0,
            "citation_quality_score": 0,
            "hallucination_score": 0,
            "hallucination_detected": True,
            "unsupported_claims": ["Evaluator returned invalid JSON."],
            "short_reason": response.content[:500],
        }

    normalized = normalize_evaluation(parsed)

    return {
        "id": record.get("id"),
        "category": record.get("category"),
        "question": question,
        "answer": answer,
        "latency_seconds": record.get("latency_seconds"),
        "retrieval_k": (
            record.get("k")
            or record.get("retrieval_final_k")
            or record.get("validated_final_k")
        ),
        "mode": record.get("mode"),
        "context_source_field": context_source_field,
        "context_items_used": len(retrieved_context),
        "evaluation": normalized,
    }


def safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0

    return round(statistics.mean(values), 3)


def summarize_evaluations(evaluated_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    score_fields = [
        "faithfulness_score",
        "answer_relevance_score",
        "context_relevance_score",
        "completeness_score",
        "citation_quality_score",
        "hallucination_score",
    ]

    summary: Dict[str, Any] = {
        "total_evaluated": len(evaluated_records),
        "overall": {},
        "by_category": {},
        "hallucination_detected_count": 0,
        "hallucination_rate": 0.0,
    }

    for field in score_fields:
        values = [
            record["evaluation"][field]
            for record in evaluated_records
            if record["evaluation"].get(field, 0) > 0
        ]
        summary["overall"][f"avg_{field}"] = safe_mean(values)

    hallucination_count = sum(
        1
        for record in evaluated_records
        if record["evaluation"].get("hallucination_detected") is True
    )

    summary["hallucination_detected_count"] = hallucination_count

    if evaluated_records:
        summary["hallucination_rate"] = round(
            hallucination_count / len(evaluated_records),
            3,
        )

    categories = sorted(
        set(record.get("category", "unknown") for record in evaluated_records)
    )

    for category in categories:
        category_records = [
            record for record in evaluated_records
            if record.get("category") == category
        ]

        category_summary = {
            "count": len(category_records),
        }

        for field in score_fields:
            values = [
                record["evaluation"][field]
                for record in category_records
                if record["evaluation"].get(field, 0) > 0
            ]
            category_summary[f"avg_{field}"] = safe_mean(values)

        category_hallucinations = sum(
            1
            for record in category_records
            if record["evaluation"].get("hallucination_detected") is True
        )

        category_summary["hallucination_detected_count"] = category_hallucinations
        category_summary["hallucination_rate"] = round(
            category_hallucinations / len(category_records),
            3,
        ) if category_records else 0.0

        summary["by_category"][category] = category_summary

    return summary


def run_evaluation(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    limit: Optional[int] = None,
) -> None:
    setup_logging()

    records = read_jsonl(input_path)

    if limit is not None:
        records = records[:limit]

    logger.info("Loaded baseline outputs: %s", len(records))

    evaluator_llm = initialize_llm(temperature=0.0)

    evaluated_records = []

    for idx, record in enumerate(records, start=1):
        logger.info(
            "Evaluating %s/%s | id=%s | category=%s",
            idx,
            len(records),
            record.get("id"),
            record.get("category"),
        )

        try:
            evaluated = evaluate_one_record(
                record=record,
                evaluator_llm=evaluator_llm,
            )
            evaluated_records.append(evaluated)

        except Exception as exc:
            logger.exception(
                "Evaluation failed for id=%s | error=%s",
                record.get("id"),
                exc,
            )

            evaluated_records.append(
                {
                    "id": record.get("id"),
                    "category": record.get("category"),
                    "question": record.get("question"),
                    "answer": record.get("answer"),
                    "latency_seconds": record.get("latency_seconds"),
                    "retrieval_k": record.get("k"),
                    "evaluation": {
                        "faithfulness_score": 0,
                        "answer_relevance_score": 0,
                        "context_relevance_score": 0,
                        "completeness_score": 0,
                        "citation_quality_score": 0,
                        "hallucination_score": 0,
                        "hallucination_detected": True,
                        "unsupported_claims": [str(exc)],
                        "short_reason": "Evaluation exception.",
                    },
                }
            )

    summary = summarize_evaluations(evaluated_records)

    write_jsonl(output_path, evaluated_records)
    write_json(summary_path, summary)

    logger.info("Evaluation records saved to: %s", output_path)
    logger.info("Evaluation summary saved to: %s", summary_path)
    logger.info(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline RAG outputs using LLM-as-a-judge."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="outputs/baseline_rag/batch_results.jsonl",
        help="Path to baseline batch results JSONL.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/evaluation/baseline_eval_results.jsonl",
        help="Path to save evaluation results JSONL.",
    )

    parser.add_argument(
        "--summary",
        type=str,
        default="outputs/evaluation/baseline_eval_summary.json",
        help="Path to save evaluation summary JSON.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of records to evaluate for testing.",
    )

    args = parser.parse_args()

    run_evaluation(
        input_path=Path(args.input),
        output_path=Path(args.output),
        summary_path=Path(args.summary),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()