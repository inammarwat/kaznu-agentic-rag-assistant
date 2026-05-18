import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_QUESTIONS_PATH = Path("evaluation/complex_questions_30.json")
DEFAULT_BASELINE_RESULTS_PATH = Path("outputs/baseline_rag/baseline_complex30_results.jsonl")
DEFAULT_AGENTIC_RESULTS_PATH = Path("outputs/agentic_rag/agentic_complex30_results.jsonl")
DEFAULT_OUTPUT_PATH = Path("outputs/evaluation/human_eval/human_evaluation_template.csv")


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def index_by_id(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    indexed = {}

    for record in records:
        record_id = record.get("id")

        if record_id:
            indexed[record_id] = record

    return indexed


def source_preview(record: Dict[str, Any], max_sources: int = 3) -> str:
    """
    Build a compact source preview for human reviewers.
    Works for both baseline and agentic outputs.
    """
    sources = (
        record.get("source_summary")
        or record.get("validated_context")
        or record.get("retrieved_context")
        or []
    )

    previews = []

    for idx, item in enumerate(sources[:max_sources], start=1):
        metadata = item.get("metadata", item)

        source_name = metadata.get("source_name") or item.get("source_name") or ""
        url = metadata.get("url") or item.get("url") or ""
        page_number = metadata.get("page_number") or item.get("page_number") or ""
        content_type = metadata.get("content_type") or item.get("content_type") or ""

        preview = f"Source {idx}: {content_type} | {source_name}"

        if page_number:
            preview += f" | page={page_number}"

        if url:
            preview += f" | url={url}"

        previews.append(preview)

    return " || ".join(previews)


def build_human_eval_rows(
    questions: List[Dict[str, Any]],
    baseline_results: List[Dict[str, Any]],
    agentic_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    baseline_by_id = index_by_id(baseline_results)
    agentic_by_id = index_by_id(agentic_results)

    rows = []

    for q in questions:
        qid = q.get("id")
        question = q.get("question", "")
        category = q.get("category", "")
        query_type = q.get("query_type", "")

        baseline = baseline_by_id.get(qid, {})
        agentic = agentic_by_id.get(qid, {})

        common_fields = {
            "question_id": qid,
            "category": category,
            "query_type": query_type,
            "question": question,
        }

        rows.append(
            {
                **common_fields,
                "system": "Baseline RAG",
                "answer": baseline.get("answer", ""),
                "source_preview": source_preview(baseline),
                "human_correctness_1_5": "",
                "human_completeness_1_5": "",
                "human_clarity_1_5": "",
                "human_usefulness_1_5": "",
                "human_trustworthiness_1_5": "",
                "human_source_transparency_1_5": "",
                "hallucination_observed_yes_no": "",
                "reviewer_comments": "",
            }
        )

        rows.append(
            {
                **common_fields,
                "system": "Agentic RAG",
                "answer": agentic.get("answer", ""),
                "source_preview": source_preview(agentic),
                "human_correctness_1_5": "",
                "human_completeness_1_5": "",
                "human_clarity_1_5": "",
                "human_usefulness_1_5": "",
                "human_trustworthiness_1_5": "",
                "human_source_transparency_1_5": "",
                "hallucination_observed_yes_no": "",
                "reviewer_comments": "",
            }
        )

    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to write.")

    fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a human evaluation CSV template for Baseline RAG vs Agentic RAG."
    )

    parser.add_argument(
        "--questions",
        type=str,
        default=str(DEFAULT_QUESTIONS_PATH),
    )

    parser.add_argument(
        "--baseline-results",
        type=str,
        default=str(DEFAULT_BASELINE_RESULTS_PATH),
    )

    parser.add_argument(
        "--agentic-results",
        type=str,
        default=str(DEFAULT_AGENTIC_RESULTS_PATH),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
    )

    args = parser.parse_args()

    questions = read_json(Path(args.questions))
    baseline_results = read_jsonl(Path(args.baseline_results))
    agentic_results = read_jsonl(Path(args.agentic_results))

    rows = build_human_eval_rows(
        questions=questions,
        baseline_results=baseline_results,
        agentic_results=agentic_results,
    )

    write_csv(Path(args.output), rows)

    print(f"Human evaluation template created: {args.output}")
    print(f"Rows created: {len(rows)}")
    print("Each question has two rows: Baseline RAG and Agentic RAG.")


if __name__ == "__main__":
    main()