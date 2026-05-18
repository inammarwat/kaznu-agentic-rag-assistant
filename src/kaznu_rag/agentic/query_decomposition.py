import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from kaznu_rag.rag.llm_client import initialize_llm

logger = logging.getLogger(__name__)


QUERY_DECOMPOSITION_SYSTEM_PROMPT = """
You are a query decomposition agent for a university information RAG system.

Your task is to analyze a user question and decide whether it should be decomposed into smaller retrieval queries.

Use decomposition when the question:
- contains multiple constraints,
- asks about more than one topic,
- requires information from multiple sources,
- combines policy, tuition, admissions, visa, AI regulation, or university facts,
- requires step-by-step reasoning.

Do NOT over-decompose simple single-fact questions.

Return ONLY valid JSON using this schema:

{
  "is_complex": true or false,
  "reason": "brief reason",
  "subqueries": [
    "retrieval query 1",
    "retrieval query 2"
  ]
}

Rules:
1. Always include the original information need.
2. Use concise retrieval-oriented subqueries.
3. Maximum 5 subqueries.
4. If the question is simple, return is_complex=false and one subquery equal to the original question.
5. Do not answer the question.
""".strip()


QUERY_DECOMPOSITION_USER_PROMPT = """
User question:
{question}

Return the decomposition JSON.
""".strip()


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from an LLM response.
    Handles clean JSON and JSON embedded inside text.
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


def clean_subquery(query: str) -> str:
    return " ".join(str(query).strip().split())


def normalize_subqueries(
    question: str,
    subqueries: Any,
    max_subqueries: int,
) -> List[str]:
    if not isinstance(subqueries, list):
        return [question]

    cleaned = []

    for item in subqueries:
        query = clean_subquery(str(item))

        if not query:
            continue

        if query not in cleaned:
            cleaned.append(query)

    if not cleaned:
        cleaned = [question]

    return cleaned[:max_subqueries]


def heuristic_is_complex(question: str) -> bool:
    """
    Lightweight fallback complexity detector.
    Used if LLM decomposition fails.
    """
    q = question.lower()

    complexity_markers = [
        " and ",
        " or ",
        " compare ",
        " difference ",
        " what happens after ",
        " if ",
        " both ",
        " requirements",
        " documents",
        " fees",
        " tuition",
        " visa",
        " academic leave",
        " academic mobility",
        " plagiarism",
        " ai",
        " ethical",
    ]

    marker_hits = sum(1 for marker in complexity_markers if marker in q)

    return marker_hits >= 2 or len(question.split()) >= 18


def fallback_decomposition(question: str, max_subqueries: int = 5) -> Dict[str, Any]:
    """
    Rule-based fallback when the LLM decomposition response is invalid.
    """
    is_complex = heuristic_is_complex(question)

    if not is_complex:
        return {
            "is_complex": False,
            "reason": "The question appears to be a single-intent query.",
            "subqueries": [question],
        }

    subqueries = [question]

    q = question.lower()

    if "tuition" in q or "fee" in q or "fees" in q:
        subqueries.append("tuition fee amount degree level applicant region language")

    if "visa" in q:
        subqueries.append("visa fee visa extension fee foreign students")

    if "document" in q or "admission" in q or "applicant" in q:
        subqueries.append("foreign applicant admission required documents")

    if "academic leave" in q:
        subqueries.append("academic leave application reasons return academic difference")

    if "academic mobility" in q:
        subqueries.append("academic mobility credit transfer academic debt")

    if "ai" in q or "generative ai" in q:
        subqueries.append("AI regulation students lecturers ethical restrictions generative AI")

    return {
        "is_complex": True,
        "reason": "The question contains multiple possible information constraints.",
        "subqueries": subqueries[:max_subqueries],
    }


def decompose_query(
    question: str,
    llm=None,
    max_subqueries: int = 5,
) -> Dict[str, Any]:
    """
    Decompose a user question into retrieval-oriented subqueries.

    Parameters:
        question: User question.
        llm: Optional initialized LangChain chat model.
        max_subqueries: Maximum number of subqueries to return.

    Returns:
        {
            "original_question": str,
            "is_complex": bool,
            "reason": str,
            "subqueries": list[str],
            "decomposition_method": "llm" or "fallback"
        }
    """
    question = clean_subquery(question)

    if not question:
        raise ValueError("Question cannot be empty.")

    if llm is None:
        llm = initialize_llm(temperature=0.0)

    prompt = QUERY_DECOMPOSITION_USER_PROMPT.format(question=question)

    try:
        response = llm.invoke(
            [
                SystemMessage(content=QUERY_DECOMPOSITION_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )

        parsed = extract_json_object(response.content)

        if parsed is None:
            raise ValueError(f"Invalid decomposition JSON: {response.content}")

        is_complex = bool(parsed.get("is_complex", False))
        reason = str(parsed.get("reason", ""))

        subqueries = normalize_subqueries(
            question=question,
            subqueries=parsed.get("subqueries", [question]),
            max_subqueries=max_subqueries,
        )

        if not is_complex and len(subqueries) > 1:
            subqueries = [question]

        return {
            "original_question": question,
            "is_complex": is_complex,
            "reason": reason,
            "subqueries": subqueries,
            "decomposition_method": "llm",
        }

    except Exception as exc:
        logger.warning("LLM query decomposition failed. Using fallback. Error: %s", exc)

        fallback = fallback_decomposition(
            question=question,
            max_subqueries=max_subqueries,
        )

        return {
            "original_question": question,
            "is_complex": fallback["is_complex"],
            "reason": fallback["reason"],
            "subqueries": fallback["subqueries"],
            "decomposition_method": "fallback",
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_questions = [
        "What is the tuition fee for Law bachelor students from far abroad in English?",
        "If a foreign bachelor applicant from far abroad wants to study Law in English, what tuition fee applies, and what additional visa or medical fees are mentioned?",
        "Can students use AI tools for research, and what ethical restrictions apply?",
    ]

    llm = initialize_llm(temperature=0.0)

    for q in test_questions:
        print("\nQUESTION:", q)
        result = decompose_query(q, llm=llm)
        print(json.dumps(result, indent=2, ensure_ascii=False))