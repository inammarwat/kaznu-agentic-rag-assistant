import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, HumanMessage

from kaznu_rag.agentic.multi_query_retriever import multi_query_retrieve
from kaznu_rag.agentic.source_validator import validate_sources
from kaznu_rag.rag.llm_client import initialize_llm
from kaznu_rag.rag.retriever import format_context

logger = logging.getLogger(__name__)


AGENTIC_RAG_SYSTEM_PROMPT = """
You are an Agentic Retrieval-Augmented Generation assistant for Al-Farabi Kazakh National University.

You must answer the user's question using ONLY the validated context.

Rules:
1. Do not invent facts.
2. If a required detail is not present in the validated context, clearly say that the available sources do not provide that detail.
3. Prefer exact values from structured records, especially tuition fees.
4. For multi-part questions, answer each part separately.
5. Use concise citations in the form [Source 1], [Source 2], etc.
6. Preserve exact numbers, currency, academic year, degree level, applicant region, and language when present.
7. Do not cite rejected or unvalidated sources.
8. Keep the answer clear and useful for students or applicants.

Your answer should be grounded, transparent, and not overly verbose.
""".strip()


AGENTIC_RAG_USER_PROMPT = """
User question:
{question}

Query decomposition:
{decomposition}

Validated retrieved context:
{context}

Answer:
""".strip()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def safe_output_filename(question: str, prefix: str = "agentic_rag") -> str:
    safe = (
        question.lower()
        .replace(" ", "_")
        .replace("?", "")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "")
        .replace(",", "")
        .replace(".", "")
    )

    safe = "".join(ch for ch in safe if ch.isalnum() or ch in {"_", "-"})
    return f"{prefix}_{safe[:90]}.json"


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compact_source_summary(validated_context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Save a compact, readable source summary for reporting and dissertation analysis.
    """
    summaries = []

    for item in validated_context:
        metadata = item.get("metadata", {})

        summaries.append(
            {
                "rank": item.get("rank"),
                "validation_score": item.get("validation_score"),
                "distance_score": item.get("score"),
                "fused_score": item.get("fused_score"),
                "content_type": metadata.get("content_type"),
                "source_name": metadata.get("source_name"),
                "page_number": metadata.get("page_number"),
                "url": metadata.get("url"),
                "faculty": metadata.get("faculty"),
                "degree_level": metadata.get("degree_level"),
                "language": metadata.get("language"),
                "applicant_region": metadata.get("applicant_region"),
                "tuition_fee_kzt": metadata.get("tuition_fee_kzt"),
                "validation_reasons": item.get("validation_reasons"),
                "text_preview": item.get("text", "")[:400],
            }
        )

    return summaries


def answer_agentic_question(
    question: str,
    config_path: Path = Path("config/settings.yaml"),
    retrieval_k_per_query: int = 5,
    retrieval_final_k: int = 10,
    validated_final_k: int = 6,
    max_subqueries: int = 5,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Run the first Agentic RAG pipeline.

    Pipeline:
    1. Initialize LLM once.
    2. Decompose query.
    3. Perform multi-query retrieval.
    4. Fuse retrieved results.
    5. Validate and rerank sources.
    6. Generate grounded answer from validated context only.
    """
    start_time = time.time()

    logger.info("Initializing LLM.")
    llm = initialize_llm(temperature=temperature)

    logger.info("Running multi-query retrieval.")
    retrieval_result = multi_query_retrieve(
        question=question,
        config_path=config_path,
        retrieval_k_per_query=retrieval_k_per_query,
        final_k=retrieval_final_k,
        max_subqueries=max_subqueries,
        include_original_query=True,
        llm=llm,
    )

    logger.info("Running source validation.")
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

    logger.info("Generating final Agentic RAG answer.")

    response = llm.invoke(
        [
            SystemMessage(content=AGENTIC_RAG_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    latency_seconds = round(time.time() - start_time, 3)

    result = {
        "question": question,
        "answer": response.content,
        "mode": "agentic_rag_v1",
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

    return result


def print_agentic_result(result: Dict[str, Any]) -> None:
    print("\nANSWER:\n")
    print(result["answer"])

    print("\nDECOMPOSITION:\n")
    print(json.dumps(result["decomposition"], indent=2, ensure_ascii=False))

    print("\nSUBQUERIES USED:\n")
    for query in result["subqueries_used"]:
        print("-", query)

    print("\nVALIDATION SUMMARY:\n")
    print(json.dumps(result["validation_summary"], indent=2, ensure_ascii=False))

    print("\nVALIDATED SOURCES:\n")
    for source in result["source_summary"]:
        print(
            f"{source['rank']}. "
            f"validation={source['validation_score']} | "
            f"distance={source['distance_score']} | "
            f"type={source['content_type']} | "
            f"source={source['source_name']} | "
            f"url={source['url']}"
        )

        if source.get("tuition_fee_kzt"):
            print(
                "   tuition:",
                {
                    "faculty": source.get("faculty"),
                    "degree_level": source.get("degree_level"),
                    "language": source.get("language"),
                    "applicant_region": source.get("applicant_region"),
                    "tuition_fee_kzt": source.get("tuition_fee_kzt"),
                },
            )


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run Agentic RAG with query decomposition, multi-query retrieval, and source validation."
    )

    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="User question.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to settings YAML file.",
    )

    parser.add_argument(
        "--retrieval-k-per-query",
        type=int,
        default=5,
        help="Top-k documents retrieved per subquery.",
    )

    parser.add_argument(
        "--retrieval-final-k",
        type=int,
        default=10,
        help="Number of fused retrieved documents before validation.",
    )

    parser.add_argument(
        "--validated-final-k",
        type=int,
        default=6,
        help="Number of validated documents passed to answer generation.",
    )

    parser.add_argument(
        "--max-subqueries",
        type=int,
        default=5,
        help="Maximum number of generated subqueries.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/agentic_rag",
        help="Directory to save Agentic RAG output JSON.",
    )

    args = parser.parse_args()

    result = answer_agentic_question(
        question=args.question,
        config_path=Path(args.config),
        retrieval_k_per_query=args.retrieval_k_per_query,
        retrieval_final_k=args.retrieval_final_k,
        validated_final_k=args.validated_final_k,
        max_subqueries=args.max_subqueries,
    )

    output_dir = Path(args.output_dir)
    output_path = output_dir / safe_output_filename(args.question)

    save_json(output_path, result)

    print_agentic_result(result)
    print(f"\nSaved output: {output_path}")


if __name__ == "__main__":
    main()