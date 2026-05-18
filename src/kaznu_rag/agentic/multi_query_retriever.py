import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document

from kaznu_rag.agentic.query_decomposition import decompose_query
from kaznu_rag.rag.llm_client import initialize_llm
from kaznu_rag.rag.retriever import load_vectorstore, format_context

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def document_key(doc: Document) -> str:
    """
    Stable key for deduplication and fusion.
    Prefer chunk_id from metadata.
    """
    chunk_id = doc.metadata.get("chunk_id")

    if chunk_id:
        return str(chunk_id)

    return str(hash(doc.page_content))


def retrieve_for_subquery(
    db,
    subquery: str,
    k: int,
) -> List[Dict[str, Any]]:
    """
    Retrieve documents for one subquery.
    Chroma distance score: lower is better.
    """
    results = db.similarity_search_with_score(subquery, k=k)

    retrieved = []

    for rank, (doc, score) in enumerate(results, start=1):
        retrieved.append(
            {
                "subquery": subquery,
                "rank": rank,
                "score": float(score),
                "doc": doc,
                "key": document_key(doc),
            }
        )

    return retrieved


def reciprocal_rank_fusion(
    all_results: List[Dict[str, Any]],
    final_k: int,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Fuse retrieval results from multiple subqueries using Reciprocal Rank Fusion.

    RRF score:
        sum(1 / (rrf_k + rank))

    Higher fused_score is better.
    """
    fused: Dict[str, Dict[str, Any]] = {}

    for item in all_results:
        key = item["key"]
        rank = item["rank"]
        score = item["score"]
        subquery = item["subquery"]
        doc = item["doc"]

        rrf_score = 1.0 / (rrf_k + rank)

        if key not in fused:
            fused[key] = {
                "doc": doc,
                "best_distance_score": score,
                "fused_score": rrf_score,
                "matched_subqueries": [subquery],
                "subquery_ranks": {subquery: rank},
            }
        else:
            fused[key]["fused_score"] += rrf_score
            fused[key]["best_distance_score"] = min(
                fused[key]["best_distance_score"],
                score,
            )

            if subquery not in fused[key]["matched_subqueries"]:
                fused[key]["matched_subqueries"].append(subquery)

            fused[key]["subquery_ranks"][subquery] = rank

    ranked_items = sorted(
        fused.values(),
        key=lambda x: (
            x["fused_score"],
            -1.0 * x["best_distance_score"],
        ),
        reverse=True,
    )

    final_results = []

    for final_rank, item in enumerate(ranked_items[:final_k], start=1):
        doc = item["doc"]

        metadata = dict(doc.metadata)
        metadata["matched_subqueries"] = json.dumps(
            item["matched_subqueries"],
            ensure_ascii=False,
        )
        metadata["subquery_ranks"] = json.dumps(
            item["subquery_ranks"],
            ensure_ascii=False,
        )
        metadata["fusion_method"] = "reciprocal_rank_fusion"

        final_results.append(
            {
                "rank": final_rank,
                "score": float(item["best_distance_score"]),
                "fused_score": round(float(item["fused_score"]), 6),
                "text": doc.page_content,
                "metadata": metadata,
            }
        )

    return final_results


def multi_query_retrieve(
    question: str,
    config_path: Path = Path("config/settings.yaml"),
    retrieval_k_per_query: int = 5,
    final_k: int = 8,
    max_subqueries: int = 5,
    include_original_query: bool = True,
    llm=None,
) -> Dict[str, Any]:
    """
    Agentic multi-query retrieval.

    Steps:
    1. Decompose question.
    2. Retrieve for each subquery.
    3. Fuse results with RRF.
    4. Return final context.
    """
    if llm is None:
        llm = initialize_llm(temperature=0.0)

    db = load_vectorstore(config_path)

    decomposition = decompose_query(
        question=question,
        llm=llm,
        max_subqueries=max_subqueries,
    )

    subqueries = decomposition["subqueries"]

    if include_original_query and question not in subqueries:
        subqueries = [question] + subqueries

    # Deduplicate while preserving order
    unique_subqueries = []
    for query in subqueries:
        if query not in unique_subqueries:
            unique_subqueries.append(query)

    logger.info("Subqueries used for retrieval: %s", unique_subqueries)

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
        final_k=final_k,
    )

    context = format_context(fused_results)

    return {
        "question": question,
        "decomposition": decomposition,
        "subqueries_used": unique_subqueries,
        "retrieved_context": fused_results,
        "formatted_context": context,
        "retrieval_k_per_query": retrieval_k_per_query,
        "final_k": final_k,
        "mode": "agentic_multi_query_retrieval",
    }


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Test agentic multi-query retrieval."
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
    )

    parser.add_argument(
        "--retrieval-k-per-query",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--final-k",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    llm = initialize_llm(temperature=0.0)

    result = multi_query_retrieve(
        question=args.question,
        config_path=Path(args.config),
        retrieval_k_per_query=args.retrieval_k_per_query,
        final_k=args.final_k,
        llm=llm,
    )

    print("\nDECOMPOSITION:\n")
    print(json.dumps(result["decomposition"], indent=2, ensure_ascii=False))

    print("\nSUBQUERIES USED:\n")
    for query in result["subqueries_used"]:
        print("-", query)

    print("\nRETRIEVED RESULTS:\n")
    for item in result["retrieved_context"]:
        metadata = item["metadata"]

        print("\n--- RESULT", item["rank"], "---")
        print("distance_score:", item["score"])
        print("fused_score:", item["fused_score"])
        print("content_type:", metadata.get("content_type"))
        print("source:", metadata.get("source_name"))
        print("page:", metadata.get("page_number"))
        print("url:", metadata.get("url"))
        print("matched_subqueries:", metadata.get("matched_subqueries"))
        print(item["text"][:700])


if __name__ == "__main__":
    main()