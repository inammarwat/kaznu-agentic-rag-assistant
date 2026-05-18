import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from kaznu_rag.agentic.agentic_rag import (
    AGENTIC_RAG_SYSTEM_PROMPT,
    AGENTIC_RAG_USER_PROMPT,
    compact_source_summary,
)
from kaznu_rag.agentic.multi_query_retriever import (
    retrieve_for_subquery,
    reciprocal_rank_fusion,
)
from kaznu_rag.agentic.query_decomposition import decompose_query
from kaznu_rag.agentic.source_validator import validate_sources
from kaznu_rag.rag.llm_client import initialize_llm
from kaznu_rag.rag.retriever import load_vectorstore, format_context

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def read_questions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError("Questions file must contain a JSON list.")

    return data


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_agentic_retrieval_with_loaded_db(
    question: str,
    db,
    llm,
    retrieval_k_per_query: int = 5,
    retrieval_final_k: int = 10,
    max_subqueries: int = 5,
    include_original_query: bool = True,
) -> Dict[str, Any]:
    """
    Agentic retrieval using an already-loaded vector store and LLM.

    This avoids reloading Chroma and reinitializing the LLM for every question.
    """
    decomposition = decompose_query(
        question=question,
        llm=llm,
        max_subqueries=max_subqueries,
    )

    subqueries = decomposition["subqueries"]

    if include_original_query and question not in subqueries:
        subqueries = [question] + subqueries

    unique_subqueries = []
    for subquery in subqueries:
        if subquery not in unique_subqueries:
            unique_subqueries.append(subquery)

    all_results = []

    for subquery in unique_subqueries:
        subquery_results = retrieve_for_subquery(
            db=db,
            subquery=subquery,
            k=retrieval_k_per_query,
        )
        all_results.extend(subquery_results)

    fused_results = reciprocal_rank_fusion(
        all_results=all_results,
        final_k=retrieval_final_k,
    )

    return {
        "question": question,
        "decomposition": decomposition,
        "subqueries_used": unique_subqueries,
        "retrieved_context": fused_results,
        "retrieval_k_per_query": retrieval_k_per_query,
        "retrieval_final_k": retrieval_final_k,
        "mode": "agentic_multi_query_retrieval_batch",
    }


def answer_one_agentic_question(
    question_item: Dict[str, Any],
    db,
    llm,
    retrieval_k_per_query: int,
    retrieval_final_k: int,
    validated_final_k: int,
    max_subqueries: int,
) -> Dict[str, Any]:
    question_id = question_item.get("id", "")
    category = question_item.get("category", "unknown")
    question = question_item["question"]

    start_time = time.time()

    retrieval_result = build_agentic_retrieval_with_loaded_db(
        question=question,
        db=db,
        llm=llm,
        retrieval_k_per_query=retrieval_k_per_query,
        retrieval_final_k=retrieval_final_k,
        max_subqueries=max_subqueries,
        include_original_query=True,
    )

    validation_result = validate_sources(
        question=question,
        retrieved_context=retrieval_result["retrieved_context"],
        final_k=validated_final_k,
        fallback_if_empty=True,
    )

    validated_context = validation_result["validated_context"]
    context_text = format_context(validated_context)

    decomposition_text = json.dumps(
        retrieval_result["decomposition"],
        ensure_ascii=False,
        indent=2,
    )

    user_prompt = AGENTIC_RAG_USER_PROMPT.format(
        question=question,
        decomposition=decomposition_text,
        context=context_text,
    )

    response = llm.invoke(
        [
            SystemMessage(content=AGENTIC_RAG_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    latency_seconds = round(time.time() - start_time, 3)

    return {
        "id": question_id,
        "category": category,
        "question": question,
        "answer": response.content,
        "mode": "agentic_rag_batch_v1",
        "latency_seconds": latency_seconds,
        "retrieval_k_per_query": retrieval_k_per_query,
        "retrieval_final_k": retrieval_final_k,
        "validated_final_k": validated_final_k,
        "decomposition": retrieval_result["decomposition"],
        "subqueries_used": retrieval_result["subqueries_used"],
        "validation_summary": validation_result["validation_summary"],
        "source_summary": compact_source_summary(validated_context),
        "validated_context": validated_context,
        "rejected_context": validation_result["rejected_context"],
    }


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    successful = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]

    category_counts: Dict[str, int] = {}
    category_latency: Dict[str, List[float]] = {}

    latencies = []

    complex_count = 0
    total_subqueries = 0
    total_validated_sources = 0
    total_rejected_sources = 0

    for item in successful:
        category = item.get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1

        latency = item.get("latency_seconds")
        if latency is not None:
            latencies.append(latency)
            category_latency.setdefault(category, []).append(latency)

        decomposition = item.get("decomposition", {})
        if decomposition.get("is_complex"):
            complex_count += 1

        total_subqueries += len(item.get("subqueries_used", []))

        validation_summary = item.get("validation_summary", {})
        total_validated_sources += validation_summary.get("kept_count", 0)
        total_rejected_sources += validation_summary.get("rejected_count", 0)

    by_category = {}

    for category, count in category_counts.items():
        cat_latencies = category_latency.get(category, [])

        by_category[category] = {
            "count": count,
            "avg_latency_seconds": round(sum(cat_latencies) / len(cat_latencies), 3)
            if cat_latencies
            else 0,
            "min_latency_seconds": min(cat_latencies) if cat_latencies else 0,
            "max_latency_seconds": max(cat_latencies) if cat_latencies else 0,
        }

    return {
        "total_questions": len(results),
        "successful_questions": len(successful),
        "failed_questions": len(failed),
        "error_rate": round(len(failed) / len(results), 3) if results else 0,
        "category_counts": category_counts,
        "by_category": by_category,
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 3)
        if latencies
        else 0,
        "min_latency_seconds": min(latencies) if latencies else 0,
        "max_latency_seconds": max(latencies) if latencies else 0,
        "complex_questions_detected": complex_count,
        "avg_subqueries_per_question": round(total_subqueries / len(successful), 3)
        if successful
        else 0,
        "avg_validated_sources_per_question": round(
            total_validated_sources / len(successful), 3
        )
        if successful
        else 0,
        "avg_rejected_sources_per_question": round(
            total_rejected_sources / len(successful), 3
        )
        if successful
        else 0,
    }


def run_batch_agentic(
    questions_path: Path,
    output_path: Path,
    summary_path: Path,
    config_path: Path,
    retrieval_k_per_query: int,
    retrieval_final_k: int,
    validated_final_k: int,
    max_subqueries: int,
    limit: Optional[int] = None,
) -> None:
    setup_logging()

    questions = read_questions(questions_path)

    if limit is not None:
        questions = questions[:limit]

    logger.info("Loaded agentic questions: %s", len(questions))

    logger.info("Loading vector store once.")
    db = load_vectorstore(config_path)

    logger.info("Initializing LLM once.")
    llm = initialize_llm(temperature=0.0)

    results = []

    for idx, question_item in enumerate(questions, start=1):
        logger.info(
            "Running agentic question %s/%s | id=%s | category=%s",
            idx,
            len(questions),
            question_item.get("id"),
            question_item.get("category"),
        )

        try:
            result = answer_one_agentic_question(
                question_item=question_item,
                db=db,
                llm=llm,
                retrieval_k_per_query=retrieval_k_per_query,
                retrieval_final_k=retrieval_final_k,
                validated_final_k=validated_final_k,
                max_subqueries=max_subqueries,
            )

            results.append(result)

            logger.info(
                "Completed id=%s | latency=%s sec | kept_sources=%s | rejected_sources=%s",
                result["id"],
                result["latency_seconds"],
                result["validation_summary"].get("kept_count"),
                result["validation_summary"].get("rejected_count"),
            )

        except Exception as exc:
            logger.exception(
                "Agentic question failed id=%s | error=%s",
                question_item.get("id"),
                exc,
            )

            results.append(
                {
                    "id": question_item.get("id"),
                    "category": question_item.get("category"),
                    "question": question_item.get("question"),
                    "answer": "",
                    "mode": "agentic_rag_batch_v1",
                    "latency_seconds": None,
                    "retrieval_k_per_query": retrieval_k_per_query,
                    "retrieval_final_k": retrieval_final_k,
                    "validated_final_k": validated_final_k,
                    "decomposition": {},
                    "subqueries_used": [],
                    "validation_summary": {},
                    "source_summary": [],
                    "validated_context": [],
                    "rejected_context": [],
                    "error": str(exc),
                }
            )

    summary = summarize_results(results)

    write_jsonl(output_path, results)
    write_json(summary_path, summary)

    logger.info("Agentic batch results saved to: %s", output_path)
    logger.info("Agentic batch summary saved to: %s", summary_path)
    logger.info(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch Agentic RAG tests."
    )

    parser.add_argument(
        "--questions",
        type=str,
        required=True,
        help="Path to JSON file containing agentic test questions.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/agentic_rag/agentic_batch_results.jsonl",
        help="Path to save Agentic RAG batch results.",
    )

    parser.add_argument(
        "--summary",
        type=str,
        default="outputs/agentic_rag/agentic_batch_summary.json",
        help="Path to save Agentic RAG batch summary.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to settings YAML.",
    )

    parser.add_argument(
        "--retrieval-k-per-query",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--retrieval-final-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--validated-final-k",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--max-subqueries",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )

    args = parser.parse_args()

    run_batch_agentic(
        questions_path=Path(args.questions),
        output_path=Path(args.output),
        summary_path=Path(args.summary),
        config_path=Path(args.config),
        retrieval_k_per_query=args.retrieval_k_per_query,
        retrieval_final_k=args.retrieval_final_k,
        validated_final_k=args.validated_final_k,
        max_subqueries=args.max_subqueries,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()