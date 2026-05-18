import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from kaznu_rag.rag.llm_client import initialize_llm

logger = logging.getLogger(__name__)


FACULTY_NAMES = [
    "Biology and Biotechnology",
    "Oriental Studies",
    "High School of Economics and Business",
    "Geography and Environmental Sciences",
    "Journalism",
    "Information Technology",
    "History",
    "Mechanics and Mathematics",
    "Medicine and Healthcare",
    "Philology",
    "Philosophy and Political Science",
    "Physics and Technology",
    "International Relations",
    "Chemistry and Chemical Technology",
    "Law",
    "Al-Farabi Business School",
]


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "at",
    "is", "are", "was", "were", "what", "how", "can", "does", "do",
    "with", "from", "by", "as", "about", "mentioned", "applies",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def normalize_text(text: Any) -> str:
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def keyword_tokens(text: str) -> List[str]:
    tokens = normalize_text(text).split()
    return [token for token in tokens if token not in STOPWORDS and len(token) >= 3]


def canonical_region(value: Any) -> str:
    """
    Normalize applicant region labels.

    Handles:
    - far_abroad
    - far abroad
    - FAR ABROAD
    - cis
    - neighbouring countries
    - neighboring countries
    """
    value_norm = normalize_text(value)

    if value_norm in {"far abroad", "far_abroad"}:
        return "far_abroad"

    if "far abroad" in value_norm:
        return "far_abroad"

    if value_norm in {
        "cis",
        "neighbouring countries",
        "neighboring countries",
        "neighbouring countries cis",
        "neighboring countries cis",
    }:
        return "cis"

    if "cis" in value_norm:
        return "cis"

    if "neighbouring" in value_norm or "neighboring" in value_norm:
        return "cis"

    return value_norm


def canonical_degree(value: Any) -> str:
    """
    Normalize degree labels.
    """
    value_norm = normalize_text(value)

    if value_norm in {"master", "masters", "master s"}:
        return "masters"

    if value_norm in {"phd", "doctoral", "doctorate"}:
        return "phd"

    if value_norm in {"bachelor", "undergraduate"}:
        return "bachelor"

    return value_norm


def canonical_language(value: Any) -> str:
    """
    Normalize language labels.
    """
    value_norm = normalize_text(value)

    if value_norm in {"english", "russian", "kazakh"}:
        return value_norm

    return value_norm


def contains_word(text: str, word: str) -> bool:
    """
    Match complete words only.
    Prevents false matches such as:
    - 'ai' inside 'available'
    - 'it' inside unrelated text
    """
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def contains_phrase(text: str, phrase: str) -> bool:
    """
    Match normalized multi-word phrases.
    """
    phrase_norm = normalize_text(phrase)
    return phrase_norm in text


def detect_faculty(question: str) -> Optional[str]:
    q = normalize_text(question)

    for faculty in FACULTY_NAMES:
        if contains_phrase(q, faculty):
            return faculty

    aliases = {
        "law": "Law",
        "information technology": "Information Technology",
        "business school": "Al-Farabi Business School",
        "economics and business": "High School of Economics and Business",
        "medicine": "Medicine and Healthcare",
        "healthcare": "Medicine and Healthcare",
        "chemistry": "Chemistry and Chemical Technology",
        "biology": "Biology and Biotechnology",
        "biotechnology": "Biology and Biotechnology",
        "journalism": "Journalism",
        "history": "History",
        "oriental studies": "Oriental Studies",
        "international relations": "International Relations",
        "philology": "Philology",
        "physics": "Physics and Technology",
        "mathematics": "Mechanics and Mathematics",
    }

    for alias, faculty in aliases.items():
        if " " in alias:
            if contains_phrase(q, alias):
                return faculty
        else:
            if contains_word(q, alias):
                return faculty

    return None


def detect_degree_level(question: str) -> Optional[str]:
    q = normalize_text(question)

    if "bachelor" in q or "undergraduate" in q:
        return "bachelor"

    if "master" in q or "masters" in q or "master s" in q:
        return "masters"

    if "phd" in q or "doctoral" in q or "doctorate" in q:
        return "phd"

    return None


def detect_language(question: str) -> Optional[str]:
    q = normalize_text(question)

    if "english" in q:
        return "english"

    if "russian" in q:
        return "russian"

    if "kazakh" in q:
        return "kazakh"

    return None


def detect_applicant_region(question: str) -> Optional[str]:
    q = normalize_text(question)

    if "far abroad" in q:
        return "far_abroad"

    if "cis" in q or "neighbouring countries" in q or "neighboring countries" in q:
        return "cis"

    return None


def detect_information_needs(question: str) -> Dict[str, bool]:
    q = normalize_text(question)

    tokens = set(q.split())

    tuition_terms = {"tuition", "fee", "fees", "cost", "pay", "payment", "amount", "total"}
    visa_terms = {"visa"}
    medical_terms = {"medical", "health", "hiv", "insurance"}
    admission_terms = {"admission", "applicant", "apply", "application", "documents"}
    ai_terms = {"ai"}

    tuition = bool(tokens.intersection(tuition_terms))
    visa = bool(tokens.intersection(visa_terms))
    medical = bool(tokens.intersection(medical_terms))
    admission = bool(tokens.intersection(admission_terms))

    academic_leave = "academic leave" in q
    academic_mobility = "academic mobility" in q

    ai_policy = (
        bool(tokens.intersection(ai_terms))
        or "artificial intelligence" in q
        or "generative ai" in q
        or "chatgpt" in q
        or "chatbot" in q
        or "multi agent" in q
        or "multiagent" in q
    )

    return {
        "tuition": tuition,
        "visa": visa,
        "medical": medical,
        "admission": admission,
        "academic_leave": academic_leave,
        "academic_mobility": academic_mobility,
        "ai_policy": ai_policy,
    }


def infer_query_constraints(question: str) -> Dict[str, Any]:
    return {
        "faculty": detect_faculty(question),
        "degree_level": detect_degree_level(question),
        "language": detect_language(question),
        "applicant_region": detect_applicant_region(question),
        "information_needs": detect_information_needs(question),
    }


def metadata_value(metadata: Dict[str, Any], key: str) -> str:
    return normalize_text(metadata.get(key, ""))


def raw_metadata_value(metadata: Dict[str, Any], key: str) -> Any:
    return metadata.get(key)


def lexical_overlap_score(question: str, text: str) -> float:
    question_tokens = set(keyword_tokens(question))
    text_tokens = set(keyword_tokens(text))

    if not question_tokens:
        return 0.0

    overlap = question_tokens.intersection(text_tokens)
    return len(overlap) / len(question_tokens)


def is_medical_exam_question_not_medicine_faculty(
    question: str,
    constraints: Dict[str, Any],
) -> bool:
    needs = constraints["information_needs"]

    if not needs["medical"]:
        return False

    faculty = constraints.get("faculty")

    # If the user explicitly asks about the Medicine faculty, do not apply this guard.
    if faculty == "Medicine and Healthcare":
        return False

    return True


def validate_tuition_fact(
    question: str,
    item: Dict[str, Any],
    constraints: Dict[str, Any],
) -> Tuple[float, bool, List[str]]:
    """
    Validate structured tuition facts against explicit query constraints.

    Returns:
        validation_score, keep, reasons
    """
    metadata = item.get("metadata", {})
    text = item.get("text", "")

    reasons: List[str] = []
    score = 0.35
    hard_mismatch = False

    needs = constraints["information_needs"]

    faculty_query = constraints.get("faculty")
    degree_query = canonical_degree(constraints.get("degree_level", ""))
    language_query = canonical_language(constraints.get("language", ""))
    region_query = canonical_region(constraints.get("applicant_region", ""))

    faculty_doc = raw_metadata_value(metadata, "faculty")
    degree_doc = canonical_degree(metadata.get("degree_level", ""))
    language_doc = canonical_language(metadata.get("language", ""))
    region_doc = canonical_region(metadata.get("applicant_region", ""))

    # Guard against: "medical examination" -> "Medicine faculty tuition".
    if is_medical_exam_question_not_medicine_faculty(question, constraints):
        if normalize_text(faculty_doc) == normalize_text("Medicine and Healthcare"):
            return (
                0.05,
                False,
                [
                    "Rejected Medicine and Healthcare tuition record because the "
                    "question asks about medical/visa requirements, not the Medicine faculty."
                ],
            )

    if faculty_query:
        if normalize_text(faculty_query) == normalize_text(faculty_doc):
            score += 0.20
            reasons.append(f"Faculty matches: {faculty_query}")
        else:
            score -= 0.35
            hard_mismatch = True
            reasons.append(f"Faculty mismatch: expected {faculty_query}, got {faculty_doc}")

    if degree_query:
        if canonical_degree(degree_query) == canonical_degree(degree_doc):
            score += 0.15
            reasons.append(f"Degree level matches: {degree_query}")
        else:
            score -= 0.25
            hard_mismatch = True
            reasons.append(f"Degree mismatch: expected {degree_query}, got {degree_doc}")

    if language_query:
        if canonical_language(language_query) == canonical_language(language_doc):
            score += 0.15
            reasons.append(f"Language matches: {language_query}")
        else:
            score -= 0.20
            hard_mismatch = True
            reasons.append(f"Language mismatch: expected {language_query}, got {language_doc}")

    if region_query:
        if canonical_region(region_query) == canonical_region(region_doc):
            score += 0.15
            reasons.append(f"Applicant region matches: {region_query}")
        else:
            score -= 0.20
            hard_mismatch = True
            reasons.append(f"Region mismatch: expected {region_query}, got {region_doc}")

    if needs["tuition"]:
        score += 0.10
        reasons.append("Question has tuition/fee intent.")

    score += min(0.10, lexical_overlap_score(question, text) * 0.10)

    keep = score >= 0.45 and not hard_mismatch

    if not keep and not reasons:
        reasons.append("Tuition record did not pass validation threshold.")

    return round(max(0.0, min(score, 1.0)), 4), keep, reasons


def validate_general_source(
    question: str,
    item: Dict[str, Any],
    constraints: Dict[str, Any],
) -> Tuple[float, bool, List[str]]:
    """
    Validate non-tuition chunks using intent-aware lexical/source checks.
    """
    metadata = item.get("metadata", {})
    text = item.get("text", "")
    source_name = str(metadata.get("source_name", ""))
    url = str(metadata.get("url", ""))
    content_type = str(metadata.get("content_type", ""))

    combined = normalize_text(" ".join([text, source_name, url, content_type]))
    needs = constraints["information_needs"]

    reasons: List[str] = []
    score = 0.10

    overlap = lexical_overlap_score(question, text)
    score += min(0.30, overlap * 0.50)

    if overlap > 0:
        reasons.append(f"Lexical overlap with question: {round(overlap, 3)}")

    if needs["visa"] and "visa" in combined:
        score += 0.25
        reasons.append("Source matches visa intent.")

    if needs["medical"] and any(
        term in combined for term in ["medical", "hiv", "insurance", "health"]
    ):
        score += 0.25
        reasons.append("Source matches medical/insurance intent.")

    if needs["admission"] and any(
        term in combined for term in ["admission", "applicant", "apply", "documents"]
    ):
        score += 0.20
        reasons.append("Source matches admission/applicant intent.")

    if needs["academic_leave"] and "academic leave" in combined:
        score += 0.30
        reasons.append("Source matches academic leave intent.")

    if needs["academic_mobility"] and "academic mobility" in combined:
        score += 0.30
        reasons.append("Source matches academic mobility intent.")

    if needs["ai_policy"] and any(
        term in combined
        for term in ["artificial intelligence", "generative ai", "chatbot", "multi agent", "ai"]
    ):
        score += 0.25
        reasons.append("Source matches AI policy intent.")

    authority = metadata_value(metadata, "authority_level")
    if "official" in authority:
        score += 0.10
        reasons.append("Official source authority boost.")

    if url or source_name.endswith(".pdf"):
        score += 0.05

    keep = score >= 0.30

    if not keep and not reasons:
        reasons.append("Source did not pass validation threshold.")

    return round(max(0.0, min(score, 1.0)), 4), keep, reasons


def validate_source_item(
    question: str,
    item: Dict[str, Any],
    constraints: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = item.get("metadata", {})
    content_type = metadata.get("content_type") or item.get("content_type")

    if content_type == "tuition_fee_fact":
        score, keep, reasons = validate_tuition_fact(
            question=question,
            item=item,
            constraints=constraints,
        )
    else:
        score, keep, reasons = validate_general_source(
            question=question,
            item=item,
            constraints=constraints,
        )

    validated_item = dict(item)
    validated_metadata = dict(metadata)

    validated_metadata["validation_score"] = score
    validated_metadata["validation_keep"] = keep
    validated_metadata["validation_reasons"] = json.dumps(reasons, ensure_ascii=False)

    validated_item["metadata"] = validated_metadata
    validated_item["validation_score"] = score
    validated_item["validation_keep"] = keep
    validated_item["validation_reasons"] = reasons

    return validated_item


def validate_sources(
    question: str,
    retrieved_context: List[Dict[str, Any]],
    final_k: int = 6,
    fallback_if_empty: bool = True,
) -> Dict[str, Any]:
    """
    Validate and rerank retrieved sources.

    Returns:
        {
            "constraints": ...,
            "validated_context": [...],
            "rejected_context": [...],
            "validation_summary": ...
        }
    """
    constraints = infer_query_constraints(question)

    validated_items = [
        validate_source_item(
            question=question,
            item=item,
            constraints=constraints,
        )
        for item in retrieved_context
    ]

    kept = [item for item in validated_items if item["validation_keep"]]
    rejected = [item for item in validated_items if not item["validation_keep"]]

    kept = sorted(
        kept,
        key=lambda x: (
            x.get("validation_score", 0),
            x.get("fused_score", 0),
            -1.0 * float(x.get("score", 999)),
        ),
        reverse=True,
    )

    if not kept and fallback_if_empty:
        fallback = sorted(
            validated_items,
            key=lambda x: (
                x.get("validation_score", 0),
                x.get("fused_score", 0),
                -1.0 * float(x.get("score", 999)),
            ),
            reverse=True,
        )[:final_k]

        for item in fallback:
            item["metadata"]["validation_keep"] = True
            item["metadata"]["validation_fallback"] = True
            item["validation_keep"] = True
            item["validation_fallback"] = True

        kept = fallback
        rejected = []

    kept = kept[:final_k]

    for idx, item in enumerate(kept, start=1):
        item["rank"] = idx

    validation_summary = {
        "original_count": len(retrieved_context),
        "kept_count": len(kept),
        "rejected_count": len(rejected),
        "constraints": constraints,
    }

    return {
        "constraints": constraints,
        "validated_context": kept,
        "rejected_context": rejected,
        "validation_summary": validation_summary,
    }


def print_validation_result(result: Dict[str, Any]) -> None:
    print("\nVALIDATION SUMMARY:\n")
    print(json.dumps(result["validation_summary"], indent=2, ensure_ascii=False))

    print("\nKEPT SOURCES:\n")

    for item in result["validated_context"]:
        metadata = item.get("metadata", {})

        print("\n--- KEPT", item.get("rank"), "---")
        print("validation_score:", item.get("validation_score"))
        print("distance_score:", item.get("score"))
        print("fused_score:", item.get("fused_score"))
        print("content_type:", metadata.get("content_type"))
        print("source:", metadata.get("source_name"))
        print("page:", metadata.get("page_number"))
        print("url:", metadata.get("url"))
        print("reasons:", item.get("validation_reasons"))
        print(item.get("text", "")[:600])

    print("\nREJECTED SOURCES:\n")

    for item in result["rejected_context"][:10]:
        metadata = item.get("metadata", {})

        print("\n--- REJECTED ---")
        print("validation_score:", item.get("validation_score"))
        print("content_type:", metadata.get("content_type"))
        print("source:", metadata.get("source_name"))
        print("reasons:", item.get("validation_reasons"))
        print(item.get("text", "")[:300])


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Validate and rerank agentic retrieved sources."
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

    from kaznu_rag.agentic.multi_query_retriever import multi_query_retrieve

    llm = initialize_llm(temperature=0.0)

    retrieval_result = multi_query_retrieve(
        question=args.question,
        config_path=Path(args.config),
        retrieval_k_per_query=args.retrieval_k_per_query,
        final_k=args.retrieval_final_k,
        llm=llm,
    )

    validation_result = validate_sources(
        question=args.question,
        retrieved_context=retrieval_result["retrieved_context"],
        final_k=args.validated_final_k,
    )

    print("\nDECOMPOSITION:\n")
    print(json.dumps(retrieval_result["decomposition"], indent=2, ensure_ascii=False))

    print_validation_result(validation_result)


if __name__ == "__main__":
    main()