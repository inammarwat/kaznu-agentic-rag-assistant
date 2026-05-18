import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from kaznu_rag.agentic.query_decomposition import decompose_query
from kaznu_rag.agentic.reflection_agent import answer_with_reflection
from kaznu_rag.agentic.source_validator import infer_query_constraints
from kaznu_rag.rag.llm_client import initialize_llm
from kaznu_rag.rag.retriever import load_vectorstore, format_context

logger = logging.getLogger(__name__)


BASELINE_SYSTEM_PROMPT = """
You are a university information assistant for Al-Farabi Kazakh National University.

Answer the user's question using ONLY the retrieved context.

Rules:
1. Do not invent facts.
2. If the answer is not present in the context, say that the available sources do not provide the information.
3. Use exact numbers, dates, degree levels, applicant regions, and language labels when present.
4. Use citations in the form [Source 1], [Source 2], etc.
5. Keep the answer concise and clear.
""".strip()


BASELINE_USER_PROMPT = """
User question:
{question}

Retrieved context:
{context}

Answer:
""".strip()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def has_any(text: str, terms: List[str]) -> bool:
    text_norm = normalize_text(text)
    return any(term in text_norm for term in terms)


def is_single_fact_tuition_query(question: str) -> bool:
    """
    Detect direct structured tuition queries.

    Example routed to baseline:
        What is the tuition fee for Law bachelor students from far abroad in English?

    Example routed to agentic:
        What is the total cost including tuition, visa, medical certificate, HIV test, and insurance?
    """
    q = normalize_text(question)
    constraints = infer_query_constraints(question)
    needs = constraints["information_needs"]

    if not needs.get("tuition"):
        return False

    # Direct tuition question should not require multi-source synthesis.
    complex_markers = [
        "including",
        "total cost",
        "exact total",
        "visa",
        "medical",
        "hiv",
        "insurance",
        "compare",
        "difference",
        "additional",
        "besides tuition",
        "documents",
        "deadline",
        "after admission",
        "what else",
    ]

    if has_any(q, complex_markers):
        return False

    if constraints.get("faculty") and constraints.get("degree_level"):
        return True

    return False


def heuristic_route(question: str) -> Dict[str, Any]:
    """
    Rule-based first-pass routing.

    Returns:
        {
            "route": "baseline" or "agentic",
            "reason": str,
            "risk_flags": list[str]
        }
    """
    q = normalize_text(question)
    constraints = infer_query_constraints(question)
    needs = constraints["information_needs"]

    risk_flags = []

    if is_single_fact_tuition_query(question):
        return {
            "route": "baseline",
            "reason": "Direct single-fact tuition query with clear constraints.",
            "risk_flags": [],
        }

    complex_markers = [
        "including",
        "compare",
        "difference",
        "total",
        "exact",
        "all",
        "what else",
        "multi",
        "after",
        "later",
        "returns",
        "if ",
        "and what",
        "documents and",
        "visa",
        "medical",
        "hiv",
        "insurance",
        "plagiarism",
        "final attestation",
        "academic mobility",
        "academic leave",
        "ethical restrictions",
        "prohibited uses",
    ]

    if has_any(q, complex_markers):
        risk_flags.append("Question contains complex or multi-part markers.")

    if sum(1 for value in needs.values() if value) >= 2:
        risk_flags.append("Question involves multiple information needs.")

    if "available information provide" in q or "do the sources provide" in q:
        risk_flags.append("Question asks about evidence sufficiency or missing information.")

    if len(q.split()) >= 18:
        risk_flags.append("Question is long and likely multi-constraint.")

    if risk_flags:
        return {
            "route": "agentic",
            "reason": "Complexity/risk markers detected.",
            "risk_flags": risk_flags,
        }

    return {
        "route": "baseline",
        "reason": "Question appears simple enough for baseline RAG.",
        "risk_flags": [],
    }


def llm_route_check(question: str, llm=None) -> Dict[str, Any]:
    """
    Use the decomposition agent as a secondary routing signal.
    """
    if llm is None:
        llm = initialize_llm(temperature=0.0)

    decomposition = decompose_query(
        question=question,
        llm=llm,
        max_subqueries=5,
    )

    if decomposition.get("is_complex"):
        return {
            "route": "agentic",
            "reason": "Query decomposition agent marked the question as complex.",
            "decomposition": decomposition,
        }

    return {
        "route": "baseline",
        "reason": "Query decomposition agent marked the question as simple.",
        "decomposition": decomposition,
    }


def context_items_from_docs(results: List[Any]) -> List[Dict[str, Any]]:
    """
    Convert LangChain similarity_search_with_score results to the same structure
    expected by format_context().
    """
    context_items = []

    for rank, (doc, score) in enumerate(results, start=1):
        metadata = dict(doc.metadata)

        context_items.append(
            {
                "rank": rank,
                "score": float(score),
                "text": doc.page_content,
                "metadata": metadata,
            }
        )

    return context_items


def answer_baseline_simple(
    question: str,
    config_path: Path,
    k: int = 5,
    llm=None,
) -> Dict[str, Any]:
    """
    Lightweight baseline RAG path used by adaptive routing.
    """
    if llm is None:
        llm = initialize_llm(temperature=0.0)

    db = load_vectorstore(config_path)

    results = db.similarity_search_with_score(question, k=k)
    retrieved_context = context_items_from_docs(results)
    context_text = format_context(retrieved_context)

    prompt = BASELINE_USER_PROMPT.format(
        question=question,
        context=context_text,
    )

    response = llm.invoke(
        [
            SystemMessage(content=BASELINE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "question": question,
        "answer": response.content,
        "mode": "adaptive_baseline_rag",
        "route": "baseline",
        "retrieved_context": retrieved_context,
        "k": k,
    }


def adaptive_answer(
    question: str,
    config_path: Path = Path("config/settings.yaml"),
    baseline_k: int = 5,
    retrieval_k_per_query: int = 5,
    retrieval_final_k: int = 10,
    validated_final_k: int = 6,
    max_subqueries: int = 5,
    use_llm_router: bool = True,
) -> Dict[str, Any]:
    """
    Adaptive RAG router.

    Routing logic:
    - Direct/simple query -> Baseline RAG
    - Complex/risky/missing-information query -> Agentic RAG v2 with reflection
    """
    start_time = time.time()

    heuristic = heuristic_route(question)
    llm_route = None
    llm = None

    # Fast path: direct structured tuition query.
    # Do not call the LLM router because these queries are intentionally handled by baseline RAG.
    if is_single_fact_tuition_query(question):
        final_route = "baseline"
        route_reason = "Direct tuition fact query; baseline selected without LLM router."

    elif heuristic["route"] == "baseline" and use_llm_router:
        llm = initialize_llm(temperature=0.0)
        llm_route = llm_route_check(question, llm=llm)

        final_route = llm_route["route"]
        route_reason = llm_route["reason"]

    else:
        final_route = heuristic["route"]
        route_reason = heuristic["reason"]

    if final_route == "baseline":
        result = answer_baseline_simple(
            question=question,
            config_path=config_path,
            k=baseline_k,
            llm=llm,
        )
    else:
        result = answer_with_reflection(
            question=question,
            config_path=config_path,
            retrieval_k_per_query=retrieval_k_per_query,
            retrieval_final_k=retrieval_final_k,
            validated_final_k=validated_final_k,
            max_subqueries=max_subqueries,
        )
        result["route"] = "agentic"

    latency_seconds = round(time.time() - start_time, 3)

    result["adaptive_routing"] = {
        "selected_route": final_route,
        "route_reason": route_reason,
        "heuristic_route": heuristic,
        "llm_route": llm_route,
        "use_llm_router": use_llm_router,
    }

    result["latency_seconds"] = latency_seconds
    result["mode"] = "adaptive_rag_v1"

    return result


def print_adaptive_result(result: Dict[str, Any]) -> None:
    print("\nROUTE:\n")
    print(json.dumps(result["adaptive_routing"], indent=2, ensure_ascii=False))

    print("\nANSWER:\n")
    print(result.get("answer", ""))

    if result.get("route") == "agentic":
        print("\nSUFFICIENCY:\n")
        print(json.dumps(result.get("sufficiency", {}), indent=2, ensure_ascii=False))

        print("\nREFLECTION:\n")
        print(json.dumps(result.get("reflection", {}), indent=2, ensure_ascii=False))

        print("\nREVISED:", result.get("revised"))

    print("\nLATENCY:", result.get("latency_seconds"))


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run Adaptive RAG: simple query -> baseline, complex query -> agentic."
    )

    parser.add_argument(
        "--question",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
    )

    parser.add_argument(
        "--baseline-k",
        type=int,
        default=5,
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
        "--no-llm-router",
        action="store_true",
        help="Disable LLM routing and use only heuristic routing.",
    )

    args = parser.parse_args()

    result = adaptive_answer(
        question=args.question,
        config_path=Path(args.config),
        baseline_k=args.baseline_k,
        retrieval_k_per_query=args.retrieval_k_per_query,
        retrieval_final_k=args.retrieval_final_k,
        validated_final_k=args.validated_final_k,
        use_llm_router=not args.no_llm_router,
    )

    print_adaptive_result(result)


if __name__ == "__main__":
    main()