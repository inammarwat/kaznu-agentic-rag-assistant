import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from kaznu_rag.rag.llm_client import initialize_llm
from kaznu_rag.rag.prompt_templates import (
    BASELINE_RAG_SYSTEM_PROMPT,
    BASELINE_RAG_USER_PROMPT,
)
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


def retrieve_with_loaded_db(db, query: str, k: int) -> List[Dict[str, Any]]:
    results = db.similarity_search_with_score(query, k=k)

    retrieved = []

    for rank, (doc, score) in enumerate(results, start=1):
        retrieved.append(
            {
                "rank": rank,
                "score": float(score),
                "text": doc.page_content,
                "metadata": dict(doc.metadata),
            }
        )

    return retrieved


def run_single_question(
    question_item: Dict[str, Any],
    db,
    llm,
    k: int,
) -> Dict[str, Any]:
    question_id = question_item.get("id", "")
    category = question_item.get("category", "unknown")
    question = question_item["question"]

    start_time = time.time()

    retrieved_docs = retrieve_with_loaded_db(
        db=db,
        query=question,
        k=k,
    )

    context = format_context(retrieved_docs)

    user_prompt = BASELINE_RAG_USER_PROMPT.format(
        question=question,
        context=context,
    )

    response = llm.invoke(
        [
            SystemMessage(content=BASELINE_RAG_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    latency_seconds = round(time.time() - start_time, 3)

    return {
        "id": question_id,
        "category": category,
        "question": question,
        "answer": response.content,
        "retrieved_context": retrieved_docs,
        "k": k,
        "latency_seconds": latency_seconds,
        "mode": "baseline_rag_batch",
    }


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)

    category_counts: Dict[str, int] = {}
    latencies = []

    for item in results:
        category = item.get("category", "unknown")
        category_counts[category] = category_counts.get(category, 0) + 1
        latencies.append(item.get("latency_seconds", 0))

    return {
        "total_questions": total,
        "category_counts": category_counts,
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 3)
        if latencies
        else 0,
        "min_latency_seconds": min(latencies) if latencies else 0,
        "max_latency_seconds": max(latencies) if latencies else 0,
    }


def run_batch(
    questions_path: Path,
    output_path: Path,
    summary_path: Path,
    config_path: Path,
    k: int,
    limit: int | None = None,
) -> None:
    setup_logging()

    questions = read_questions(questions_path)

    if limit is not None:
        questions = questions[:limit]

    logger.info("Loaded questions: %s", len(questions))

    logger.info("Loading vector store once.")
    db = load_vectorstore(config_path)

    logger.info("Initializing LLM once.")
    llm = initialize_llm(temperature=0.0)

    results = []

    for idx, question_item in enumerate(questions, start=1):
        logger.info(
            "Running question %s/%s | id=%s | category=%s",
            idx,
            len(questions),
            question_item.get("id"),
            question_item.get("category"),
        )

        try:
            result = run_single_question(
                question_item=question_item,
                db=db,
                llm=llm,
                k=k,
            )
            results.append(result)

            logger.info(
                "Completed id=%s | latency=%s sec",
                result["id"],
                result["latency_seconds"],
            )

        except Exception as exc:
            logger.exception(
                "Failed question id=%s | error=%s",
                question_item.get("id"),
                exc,
            )

            results.append(
                {
                    "id": question_item.get("id"),
                    "category": question_item.get("category"),
                    "question": question_item.get("question"),
                    "answer": "",
                    "retrieved_context": [],
                    "k": k,
                    "latency_seconds": None,
                    "mode": "baseline_rag_batch",
                    "error": str(exc),
                }
            )

    summary = summarize_results(
        [r for r in results if r.get("latency_seconds") is not None]
    )

    write_jsonl(output_path, results)
    write_json(summary_path, summary)

    logger.info("Batch results saved to: %s", output_path)
    logger.info("Batch summary saved to: %s", summary_path)
    logger.info(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run batch baseline RAG evaluation questions."
    )

    parser.add_argument(
        "--questions",
        type=str,
        required=True,
        help="Path to JSON file containing test questions.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/baseline_rag/batch_results.jsonl",
        help="Path to save batch RAG outputs.",
    )

    parser.add_argument(
        "--summary",
        type=str,
        default="outputs/baseline_rag/batch_summary.json",
        help="Path to save batch summary.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to settings YAML.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of retrieved chunks.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )

    args = parser.parse_args()

    run_batch(
        questions_path=Path(args.questions),
        output_path=Path(args.output),
        summary_path=Path(args.summary),
        config_path=Path(args.config),
        k=args.k,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()