import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from kaznu_rag.agentic.agentic_rag import (
    AGENTIC_RAG_SYSTEM_PROMPT,
    AGENTIC_RAG_USER_PROMPT,
    compact_source_summary,
)
from kaznu_rag.agentic.multi_query_retriever import multi_query_retrieve
from kaznu_rag.agentic.source_validator import validate_sources
from kaznu_rag.agentic.source_sufficiency import evaluate_source_sufficiency
from kaznu_rag.rag.llm_client import initialize_llm
from kaznu_rag.rag.retriever import format_context

logger = logging.getLogger(__name__)


REFLECTION_SYSTEM_PROMPT = """
You are a reflection and critique agent for an Agentic RAG university information assistant.

Your task is to evaluate whether a generated answer is:
1. fully grounded in the validated context,
2. complete with respect to the user question,
3. consistent with the source sufficiency result,
4. free from unsupported claims,
5. clear about missing information.

Return ONLY valid JSON using this schema:

{
  "answer_is_acceptable": true or false,
  "faithfulness_score": 1 to 5,
  "completeness_score": 1 to 5,
  "missing_answer_parts": ["missing part 1"],
  "unsupported_claims": ["unsupported claim 1"],
  "revision_instructions": "specific instructions for improving the answer",
  "short_reason": "brief explanation"
}

Rules:
- Be strict.
- If source sufficiency says evidence is insufficient, the answer must not invent missing details.
- If exact values are missing, the answer must explicitly say they are not provided.
- If the answer includes unsupported numbers, dates, policies, or requirements, mark it unacceptable.
- If the answer is acceptable, revision_instructions can be empty.
""".strip()


REFLECTION_USER_PROMPT = """
User question:
{question}

Source sufficiency result:
{sufficiency}

Validated context:
{context}

Generated answer:
{answer}

Evaluate the answer.
Return JSON only.
""".strip()


REVISION_SYSTEM_PROMPT = """
You are an answer revision agent for an Agentic RAG university information assistant.

Revise the answer using ONLY the validated context and the reflection feedback.

Rules:
1. Do not invent facts.
2. Remove unsupported claims.
3. If information is missing, explicitly say the validated sources do not provide it.
4. Preserve exact supported values.
5. Use citations in the form [Source 1], [Source 2], etc.
6. Keep the answer concise and student-friendly.
""".strip()


REVISION_USER_PROMPT = """
User question:
{question}

Source sufficiency result:
{sufficiency}

Validated context:
{context}

Original answer:
{answer}

Reflection feedback:
{reflection}

Write the revised final answer.
""".strip()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


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


def normalize_reflection(parsed: Dict[str, Any]) -> Dict[str, Any]:
    faithfulness_score = parsed.get("faithfulness_score", 1)
    completeness_score = parsed.get("completeness_score", 1)

    try:
        faithfulness_score = int(faithfulness_score)
    except Exception:
        faithfulness_score = 1

    try:
        completeness_score = int(completeness_score)
    except Exception:
        completeness_score = 1

    faithfulness_score = max(1, min(faithfulness_score, 5))
    completeness_score = max(1, min(completeness_score, 5))

    missing_answer_parts = parsed.get("missing_answer_parts", [])
    unsupported_claims = parsed.get("unsupported_claims", [])

    if not isinstance(missing_answer_parts, list):
        missing_answer_parts = [str(missing_answer_parts)]

    if not isinstance(unsupported_claims, list):
        unsupported_claims = [str(unsupported_claims)]

    answer_is_acceptable = bool(parsed.get("answer_is_acceptable", False))

    if (
        faithfulness_score < 5
        or completeness_score <= 3
        or unsupported_claims
    ):
        answer_is_acceptable = False

    # if faithfulness_score < 5 or unsupported_claims:
    #     answer_is_acceptable = False

    return {
        "answer_is_acceptable": answer_is_acceptable,
        "faithfulness_score": faithfulness_score,
        "completeness_score": completeness_score,
        "missing_answer_parts": [
            str(x) for x in missing_answer_parts if str(x).strip()
        ],
        "unsupported_claims": [
            str(x) for x in unsupported_claims if str(x).strip()
        ],
        "revision_instructions": str(
            parsed.get("revision_instructions", "")
        ).strip(),
        "short_reason": str(parsed.get("short_reason", "")).strip(),
    }


def critique_answer(
    question: str,
    answer: str,
    sufficiency_result: Dict[str, Any],
    validated_context: List[Dict[str, Any]],
    llm=None,
) -> Dict[str, Any]:
    if llm is None:
        llm = initialize_llm(temperature=0.0)

    context_text = format_context(validated_context)

    prompt = REFLECTION_USER_PROMPT.format(
        question=question,
        sufficiency=json.dumps(sufficiency_result, ensure_ascii=False, indent=2),
        context=context_text,
        answer=answer,
    )

    try:
        response = llm.invoke(
            [
                SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )

        parsed = extract_json_from_text(response.content)

        if parsed is None:
            raise ValueError(f"Invalid reflection JSON: {response.content}")

        result = normalize_reflection(parsed)
        result["reflection_method"] = "llm"

        return result

    except Exception as exc:
        logger.warning("Reflection failed. Error: %s", exc)

        return {
            "answer_is_acceptable": False,
            "faithfulness_score": 1,
            "completeness_score": 1,
            "missing_answer_parts": [],
            "unsupported_claims": [
                "Reflection agent failed to parse or evaluate the answer."
            ],
            "revision_instructions": (
                "Rewrite the answer conservatively using only validated context."
            ),
            "short_reason": str(exc),
            "reflection_method": "fallback",
        }


def revise_answer(
    question: str,
    answer: str,
    reflection_result: Dict[str, Any],
    sufficiency_result: Dict[str, Any],
    validated_context: List[Dict[str, Any]],
    llm=None,
) -> str:
    if llm is None:
        llm = initialize_llm(temperature=0.0)

    context_text = format_context(validated_context)

    prompt = REVISION_USER_PROMPT.format(
        question=question,
        sufficiency=json.dumps(sufficiency_result, ensure_ascii=False, indent=2),
        context=context_text,
        answer=answer,
        reflection=json.dumps(reflection_result, ensure_ascii=False, indent=2),
    )

    response = llm.invoke(
        [
            SystemMessage(content=REVISION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    return response.content


def answer_with_reflection(
    question: str,
    config_path: Path = Path("config/settings.yaml"),
    retrieval_k_per_query: int = 5,
    retrieval_final_k: int = 10,
    validated_final_k: int = 6,
    max_subqueries: int = 5,
) -> Dict[str, Any]:
    """
    Agentic RAG v2:
    query decomposition -> multi-query retrieval -> source validation
    -> source sufficiency -> answer -> reflection -> optional revision
    """
    llm = initialize_llm(temperature=0.0)

    retrieval_result = multi_query_retrieve(
        question=question,
        config_path=config_path,
        retrieval_k_per_query=retrieval_k_per_query,
        final_k=retrieval_final_k,
        max_subqueries=max_subqueries,
        include_original_query=True,
        llm=llm,
    )

    validation_result = validate_sources(
        question=question,
        retrieved_context=retrieval_result["retrieved_context"],
        final_k=validated_final_k,
        fallback_if_empty=True,
    )

    validated_context = validation_result["validated_context"]

    sufficiency_result = evaluate_source_sufficiency(
        question=question,
        decomposition=retrieval_result["decomposition"],
        validated_context=validated_context,
        llm=llm,
    )

    context_text = format_context(validated_context)

    user_prompt = AGENTIC_RAG_USER_PROMPT.format(
        question=question,
        decomposition=json.dumps(
            retrieval_result["decomposition"],
            ensure_ascii=False,
            indent=2,
        ),
        context=context_text,
    )

    answer_response = llm.invoke(
        [
            SystemMessage(content=AGENTIC_RAG_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    initial_answer = answer_response.content

    reflection_result = critique_answer(
        question=question,
        answer=initial_answer,
        sufficiency_result=sufficiency_result,
        validated_context=validated_context,
        llm=llm,
    )

    if reflection_result["answer_is_acceptable"]:
        final_answer = initial_answer
        revised = False
    else:
        final_answer = revise_answer(
            question=question,
            answer=initial_answer,
            reflection_result=reflection_result,
            sufficiency_result=sufficiency_result,
            validated_context=validated_context,
            llm=llm,
        )
        revised = True

    return {
        "question": question,
        "mode": "agentic_rag_v2_reflection",
        "initial_answer": initial_answer,
        "final_answer": final_answer,
        "answer": final_answer,
        "revised": revised,
        "decomposition": retrieval_result["decomposition"],
        "subqueries_used": retrieval_result["subqueries_used"],
        "validation_summary": validation_result["validation_summary"],
        "sufficiency": sufficiency_result,
        "reflection": reflection_result,
        "source_summary": compact_source_summary(validated_context),
        "validated_context": validated_context,
        "rejected_context": validation_result["rejected_context"],
    }


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run Agentic RAG v2 with source sufficiency and reflection."
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

    result = answer_with_reflection(
        question=args.question,
        config_path=Path(args.config),
        retrieval_k_per_query=args.retrieval_k_per_query,
        retrieval_final_k=args.retrieval_final_k,
        validated_final_k=args.validated_final_k,
    )

    print("\nFINAL ANSWER:\n")
    print(result["final_answer"])

    print("\nSUFFICIENCY:\n")
    print(json.dumps(result["sufficiency"], indent=2, ensure_ascii=False))

    print("\nREFLECTION:\n")
    print(json.dumps(result["reflection"], indent=2, ensure_ascii=False))

    print("\nREVISED:", result["revised"])


if __name__ == "__main__":
    main()