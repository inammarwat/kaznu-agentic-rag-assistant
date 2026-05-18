import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_settings(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    if not path.exists():
        logger.warning("JSONL file does not exist: %s", path)
        return records

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def make_chunk_id(parent_doc_id: str, chunk_index: int) -> str:
    return f"{parent_doc_id}::chunk-{chunk_index}"


def get_text_splitter(
    source_type: str,
    content_type: str,
    chunk_cfg: Dict[str, Any],
) -> RecursiveCharacterTextSplitter:
    if content_type == "table":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_cfg["table_chunk_size"],
            chunk_overlap=chunk_cfg["table_chunk_overlap"],
            separators=["\n", "|", " ", ""],
        )

    if source_type == "web":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_cfg["web_chunk_size"],
            chunk_overlap=chunk_cfg["web_chunk_overlap"],
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg["pdf_chunk_size"],
        chunk_overlap=chunk_cfg["pdf_chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def get_chunk_strategy(source_type: str, content_type: str) -> str:
    if content_type == "table":
        return "table_recursive_split"

    if source_type == "web":
        return "web_recursive_character_split"

    return "pdf_recursive_character_split"


def chunk_document_record(
    record: Dict[str, Any],
    chunk_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    text = (record.get("text") or "").strip()

    if not text:
        return []

    source_type = record.get("source_type", "unknown")
    content_type = record.get("content_type", "unknown")
    metadata = record.get("metadata") or {}
    min_chunk_chars = chunk_cfg["min_chunk_chars"]

    splitter = get_text_splitter(
        source_type=source_type,
        content_type=content_type,
        chunk_cfg=chunk_cfg,
    )

    split_texts = splitter.split_text(text)

    chunks: List[Dict[str, Any]] = []

    for idx, chunk_text in enumerate(split_texts):
        chunk_text = chunk_text.strip()

        if len(chunk_text) < min_chunk_chars:
            continue

        chunks.append(
            {
                "chunk_id": make_chunk_id(record["doc_id"], idx),
                "parent_doc_id": record["doc_id"],
                "chunk_index": idx,
                "text": chunk_text,
                "source_type": source_type,
                "content_type": content_type,
                "source_name": record.get("source_name"),
                "url": record.get("url"),
                "page_number": record.get("page_number"),
                "title": record.get("title"),
                "metadata": {
                    **metadata,
                    "chunk_strategy": get_chunk_strategy(source_type, content_type),
                    "chunk_char_length": len(chunk_text),
                },
            }
        )

    return chunks


def tuition_record_to_chunk(record: Dict[str, Any], index: int) -> Dict[str, Any]:
    faculty = record["faculty"]
    applicant_region = record["applicant_region"]
    degree_level = record["degree_level"]
    language = record["language"]
    fee = record["tuition_fee_kzt"]
    year = record["academic_year"]

    text = (
        f"Tuition fee for {faculty}: "
        f"applicant region = {applicant_region}, "
        f"degree level = {degree_level}, "
        f"language = {language}, "
        f"academic year = {year}, "
        f"tuition fee = {fee} KZT."
    )

    return {
        "chunk_id": (
            f"tuition::{record['faculty_index']}::"
            f"{applicant_region}::{degree_level}::{language}"
        ),
        "parent_doc_id": f"tuition::{record['faculty_index']}",
        "chunk_index": index,
        "text": text,
        "source_type": "structured",
        "content_type": "tuition_fee_fact",
        "source_name": record.get("source_name", "TUITION_FEE_2022-2023.pdf"),
        "url": None,
        "page_number": None,
        "title": "Normalized tuition fee record",
        "metadata": {
            **record,
            "document_group": "tuition_fee",
            "authority_level": "official_policy_or_regulation",
            "chunk_strategy": "structured_fact_as_single_chunk",
            "chunk_char_length": len(text),
        },
    }


def build_chunks(settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    paths = settings["paths"]
    chunk_cfg = settings["chunking"]

    documents_path = Path(paths["processed_docs_path"])
    tuition_path = Path(paths["normalized_tuition_path"])

    document_records = read_jsonl(documents_path)
    tuition_records = read_jsonl(tuition_path)

    logger.info("Loaded document records: %s", len(document_records))
    logger.info("Loaded tuition records: %s", len(tuition_records))

    chunks: List[Dict[str, Any]] = []

    for record in document_records:
        chunks.extend(chunk_document_record(record, chunk_cfg))

    for idx, record in enumerate(tuition_records):
        chunks.append(tuition_record_to_chunk(record, idx))

    return chunks


def summarize_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_source_type: Dict[str, int] = {}
    by_content_type: Dict[str, int] = {}
    by_chunk_strategy: Dict[str, int] = {}

    for chunk in chunks:
        source_type = chunk.get("source_type", "unknown")
        content_type = chunk.get("content_type", "unknown")
        chunk_strategy = chunk.get("metadata", {}).get("chunk_strategy", "unknown")

        by_source_type[source_type] = by_source_type.get(source_type, 0) + 1
        by_content_type[content_type] = by_content_type.get(content_type, 0) + 1
        by_chunk_strategy[chunk_strategy] = by_chunk_strategy.get(chunk_strategy, 0) + 1

    lengths = [len(chunk["text"]) for chunk in chunks]

    return {
        "total_chunks": len(chunks),
        "by_source_type": by_source_type,
        "by_content_type": by_content_type,
        "by_chunk_strategy": by_chunk_strategy,
        "min_chunk_chars": min(lengths) if lengths else 0,
        "max_chunk_chars": max(lengths) if lengths else 0,
        "avg_chunk_chars": round(sum(lengths) / len(lengths), 2) if lengths else 0,
    }


def run_chunking(config_path: Path) -> None:
    setup_logging()

    settings = load_settings(config_path)
    paths = settings["paths"]

    chunks_path = Path(paths["chunks_path"])
    report_path = Path(paths["chunk_report_path"])

    chunks = build_chunks(settings)
    report = summarize_chunks(chunks)

    write_jsonl(chunks_path, chunks)
    write_json(report_path, report)

    logger.info("Chunks saved to: %s", chunks_path)
    logger.info("Chunk report saved to: %s", report_path)
    logger.info(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk processed KazNU documents for vector store creation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to settings YAML file.",
    )
    args = parser.parse_args()

    run_chunking(Path(args.config))


if __name__ == "__main__":
    main()