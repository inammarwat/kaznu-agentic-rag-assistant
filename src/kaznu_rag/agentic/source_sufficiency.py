import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from kaznu_rag.agentic.multi_query_retriever import multi_query_retrieve
from kaznu_rag.agentic.source_validator import validate_sources
from kaznu_rag.rag.llm_client import initialize_llm
from kaznu_rag.rag.retriever import format_context

logger = logging.getLogger(__name__)


SOURCE_SUFFICIENCY_SYSTEM_PROMPT = """
You are a source sufficiency evaluator for an Agentic RAG university information assistant.

Your task is to decide whether the validated retrieved context is sufficient to answer the user's question.

You must not answer the user question. You only judge evidence sufficiency.

Evaluate:
1. Does the context cover all parts of the question?
2. Are exact numbers, fees, dates, policies, requirements, or procedures present when needed?
3. Are there missing details that would make a complete answer impossible?
4. Are there risks of unsupported claims?
5. Should the final answer be normal, cautious, or refuse unsupported details?

Return ONLY valid JSON using this schema:

{
  "sufficient": true or false,
  "coverage_score": 1 to 5,
  "confidence_score": 1 to 5,
  "missing_information": ["missing item 1", "missing item 2"],
  "supported_information": ["supported item 1", "supported item 2"],
  "risk_flags": ["risk 1", "risk 2"],
  "recommended_answer_mode": "normal" or "cautious" or "insufficient_evidence",
  "short_reason": "brief explanation"
}

Scoring guide:
- coverage_score 5: all parts are clearly supported
- coverage_score 4: almost complete; minor missing detail
- coverage_score 3: partially supported
- coverage_score 2: weak support
- coverage_score 1: mostly unsupported

Rules:
- Be strict about exact monetary amounts, deadlines, policies, and requirements.
- If the user asks for an exact value and the context does not contain it, mark that item as missing.
- If the context supports only part of the question, sufficient=false.
- Do not invent missing details.
""".strip()


SOURCE_SUFFICIENCY_USER_PROMPT = """
User question:
{question}

Query decomposition:
{decomposition}

Validated context:
{context}

Evaluate whether the validated context is sufficient.
Return JSON only.
""".strip()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract a JSON object from an LLM response.
    """
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


def normalize_sufficiency_result(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize evaluator output so downstream modules can rely on stable fields.
    """
    coverage_score = parsed.get("coverage_score", 1)
    confidence_score = parsed.get("confidence_score", 1)

    try:
        coverage_score = int(coverage_score)
    except Exception:
        coverage_score = 1

    try:
        confidence_score = int(confidence_score)
    except Exception:
        confidence_score = 1

    coverage_score = max(1, min(coverage_score, 5))
    confidence_score = max(1, min(confidence_score, 5))

    recommended_answer_mode = str(
        parsed.get("recommended_answer_mode", "cautious")
    ).strip()

    if recommended_answer_mode not in {
        "normal",
        "cautious",
        "insufficient_evidence",
    }:
        recommended_answer_mode = "cautious"

    missing_information = parsed.get("missing_information", [])
    supported_information = parsed.get("supported_information", [])
    risk_flags = parsed.get("risk_flags", [])

    if not isinstance(missing_information, list):
        missing_information = [str(missing_information)]

    if not isinstance(supported_information, list):
        supported_information = [str(supported_information)]

    if not isinstance(risk_flags, list):
        risk_flags = [str(risk_flags)]

    sufficient = bool(parsed.get("sufficient", False))

    # Conservative correction:
    # If coverage is below 5 or missing information exists, do not mark fully sufficient.
    if coverage_score < 5 or missing_information:
        sufficient = False

    return {
        "sufficient": sufficient,
        "coverage_score": coverage_score,
        "confidence_score": confidence_score,
        "missing_information": [str(x) for x in missing_information if str(x).strip()],
        "supported_information": [str(x) for x in supported_information if str(x).strip()],
        "risk_flags": [str(x) for x in risk_flags if str(x).strip()],
        "recommended_answer_mode": recommended_answer_mode,
        "short_reason": str(parsed.get("short_reason", "")).strip(),
    }


def fallback_sufficiency_check(
    question: str,
    validated_context: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Fallback if the LLM returns invalid JSON.

    This is intentionally conservative.
    """
    context_text = " ".join(item.get("text", "") for item in validated_context).lower()
    q = question.lower()

    missing = []
    risks = []

    exact_value_terms = [
        "exact",
        "fee",
        "fees",
        "cost",
        "deadline",
        "amount",
        "total",
        "penalty",
        "penalties",
    ]

    asks_exact = any(term in q for term in exact_value_terms)

    if asks_exact:
        value_markers = ["kzt", "tenge", "$", "fee", "cost", "deadline", "date"]
        has_value_marker = any(marker in context_text for marker in value_markers)

        if not has_value_marker:
            missing.append("The question asks for exact values, but the context may not contain exact values.")

    if not validated_context:
        missing.append("No validated context is available.")
        risks.append("Answer would require unsupported generation.")

    coverage_score = 2 if missing else 4
    confidence_score = 2 if missing else 3

    return {
        "sufficient": False if missing else True,
        "coverage_score": coverage_score,
        "confidence_score": confidence_score,
        "missing_information": missing,
        "supported_information": [],
        "risk_flags": risks,
        "recommended_answer_mode": "insufficient_evidence" if missing else "cautious",
        "short_reason": "Fallback sufficiency check was used because LLM JSON parsing failed.",
    }


def evaluate_source_sufficiency(
    question: str,
    decomposition: Dict[str, Any],
    validated_context: List[Dict[str, Any]],
    llm=None,
) -> Dict[str, Any]:
    """
    Evaluate whether validated sources are enough to answer the question.

    Returns:
        {
            "sufficient": bool,
            "coverage_score": int,
            "confidence_score": int,
            "missing_information": list[str],
            "supported_information": list[str],
            "risk_flags": list[str],
            "recommended_answer_mode": str,
            "short_reason": str
        }
    """
    if llm is None:
        llm = initialize_llm(temperature=0.0)

    context_text = format_context(validated_context)

    decomposition_text = json.dumps(
        decomposition,
        ensure_ascii=False,
        indent=2,
    )

    prompt = SOURCE_SUFFICIENCY_USER_PROMPT.format(
        question=question,
        decomposition=decomposition_text,
        context=context_text,
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=SOURCE_SUFFICIENCY_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )

        parsed = extract_json_from_text(response.content)

        if parsed is None:
            raise ValueError(f"Invalid sufficiency JSON: {response.content}")

        normalized = normalize_sufficiency_result(parsed)
        normalized["sufficiency_method"] = "llm"

        return normalized

    except Exception as exc:
        logger.warning("Source sufficiency LLM check failed. Error: %s", exc)

        fallback = fallback_sufficiency_check(
            question=question,
            validated_context=validated_context,
        )
        fallback["sufficiency_method"] = "fallback"

        return fallback


def run_sufficiency_test(
    question: str,
    config_path: Path,
    retrieval_k_per_query: int,
    retrieval_final_k: int,
    validated_final_k: int,
) -> Dict[str, Any]:
    """
    End-to-end test:
    question -> multi-query retrieval -> validation -> sufficiency scoring.
    """
    llm = initialize_llm(temperature=0.0)

    retrieval_result = multi_query_retrieve(
        question=question,
        config_path=config_path,
        retrieval_k_per_query=retrieval_k_per_query,
        final_k=retrieval_final_k,
        llm=llm,
    )

    validation_result = validate_sources(
        question=question,
        retrieved_context=retrieval_result["retrieved_context"],
        final_k=validated_final_k,
        fallback_if_empty=True,
    )

    sufficiency_result = evaluate_source_sufficiency(
        question=question,
        decomposition=retrieval_result["decomposition"],
        validated_context=validation_result["validated_context"],
        llm=llm,
    )

    return {
        "question": question,
        "decomposition": retrieval_result["decomposition"],
        "subqueries_used": retrieval_result["subqueries_used"],
        "validation_summary": validation_result["validation_summary"],
        "sufficiency": sufficiency_result,
        "validated_sources": [
            {
                "rank": item.get("rank"),
                "validation_score": item.get("validation_score"),
                "content_type": item.get("metadata", {}).get("content_type"),
                "source_name": item.get("metadata", {}).get("source_name"),
                "url": item.get("metadata", {}).get("url"),
                "page_number": item.get("metadata", {}).get("page_number"),
                "text_preview": item.get("text", "")[:300],
            }
            for item in validation_result["validated_context"]
        ],
    }


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Test source sufficiency scoring for Agentic RAG."
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

    args = parser.parse_args()

    result = run_sufficiency_test(
        question=args.question,
        config_path=Path(args.config),
        retrieval_k_per_query=args.retrieval_k_per_query,
        retrieval_final_k=args.retrieval_final_k,
        validated_final_k=args.validated_final_k,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()