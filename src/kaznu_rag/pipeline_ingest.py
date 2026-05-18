import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

from kaznu_rag.ingest.pdf_loader import (
    extract_pdf_text_records,
    extract_pdf_table_records,
)
from kaznu_rag.ingest.web_loader import extract_web_records, read_urls
from kaznu_rag.utils import ensure_dirs, write_jsonl, write_json, setup_logging

logger = logging.getLogger(__name__)


def load_settings(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    if not settings:
        raise ValueError(f"Config file is empty or invalid: {config_path}")

    return settings


def validate_required_paths(paths: Dict[str, str]) -> None:
    """Validate required input paths before ingestion."""
    raw_pdf_dir = Path(paths["raw_pdf_dir"])
    urls_file = Path(paths["urls_file"])

    if not raw_pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {raw_pdf_dir}")

    if not urls_file.exists():
        raise FileNotFoundError(f"URLs file not found: {urls_file}")


def build_report(
    pdf_files: List[Path],
    valid_urls_count: int,
    all_records: list,
    all_table_records: list,
    table_manifest: list,
    processed_docs_path: Path,
    processed_tables_path: Path,
    ingestion_report_path: Path,
) -> Dict[str, Any]:
    """Create reproducible ingestion summary report."""
    pdf_text_records = [
        r for r in all_records
        if r.source_type == "pdf" and r.content_type == "text"
    ]

    pdf_table_records = [
        r for r in all_records
        if r.source_type == "pdf" and r.content_type == "table"
    ]

    web_records = [
        r for r in all_records
        if r.source_type == "web"
    ]

    return {
        "pdf_files_found": len(pdf_files),
        "web_urls_found": valid_urls_count,
        "records_total": len(all_records),
        "pdf_text_records": len(pdf_text_records),
        "pdf_table_records": len(pdf_table_records),
        "web_records": len(web_records),
        "tables_extracted_count": len(table_manifest),
        "tables_extracted": table_manifest,
        "outputs": {
            "documents_jsonl": str(processed_docs_path),
            "tables_jsonl": str(processed_tables_path),
            "ingestion_report": str(ingestion_report_path),
        },
    }


def run_ingestion(config_path: Path) -> None:
    """Run full Stage 1 ingestion: PDFs, PDF tables, and web pages."""
    settings = load_settings(config_path)

    paths = settings["paths"]
    ingestion_cfg = settings["ingestion"]

    setup_logging(paths["log_path"])
    validate_required_paths(paths)

    raw_pdf_dir = Path(paths["raw_pdf_dir"])
    urls_file = Path(paths["urls_file"])
    html_dir = Path(paths["interim_html_dir"])
    tables_dir = Path(paths["interim_tables_dir"])

    processed_docs_path = Path(paths["processed_docs_path"])
    processed_tables_path = Path(paths["processed_tables_path"])
    ingestion_report_path = Path(paths["ingestion_report_path"])

    ensure_dirs(
        html_dir,
        tables_dir,
        processed_docs_path.parent,
        processed_tables_path.parent,
        ingestion_report_path.parent,
        Path(paths["log_path"]).parent,
    )

    all_records = []
    all_table_records = []
    table_manifest = []

    pdf_files = sorted(raw_pdf_dir.glob("*.pdf"))
    valid_urls = read_urls(urls_file)

    logger.info("Starting ingestion pipeline.")
    logger.info("PDF files found: %s", len(pdf_files))
    logger.info("Valid URLs found: %s", len(valid_urls))

    # PDF text + table extraction
    for pdf_path in pdf_files:
        logger.info("Processing PDF: %s", pdf_path)

        try:
            text_records = extract_pdf_text_records(
                pdf_path=pdf_path,
                min_text_chars=ingestion_cfg["min_text_chars"],
            )
            all_records.extend(text_records)

            table_records, pdf_table_manifest = extract_pdf_table_records(
                pdf_path=pdf_path,
                output_table_dir=tables_dir,
            )
            all_records.extend(table_records)
            all_table_records.extend(table_records)
            table_manifest.extend(pdf_table_manifest)

            logger.info(
                "PDF processed: %s | text_records=%s | table_records=%s",
                pdf_path.name,
                len(text_records),
                len(table_records),
            )

        except Exception as exc:
            logger.exception("PDF ingestion failed for %s | error=%s", pdf_path, exc)

    # Web extraction
    logger.info("Starting web ingestion.")

    try:
        web_records = extract_web_records(
            urls_file=urls_file,
            html_output_dir=html_dir,
            timeout_seconds=ingestion_cfg["request_timeout_seconds"],
            user_agent=ingestion_cfg["user_agent"],
            min_text_chars=ingestion_cfg["min_text_chars"],
        )
        all_records.extend(web_records)

        logger.info("Web ingestion complete. web_records=%s", len(web_records))

    except Exception as exc:
        logger.exception("Web ingestion failed | error=%s", exc)

    # Persist outputs
    write_jsonl(processed_docs_path, all_records)
    write_jsonl(processed_tables_path, all_table_records)

    report = build_report(
        pdf_files=pdf_files,
        valid_urls_count=len(valid_urls),
        all_records=all_records,
        all_table_records=all_table_records,
        table_manifest=table_manifest,
        processed_docs_path=processed_docs_path,
        processed_tables_path=processed_tables_path,
        ingestion_report_path=ingestion_report_path,
    )

    write_json(ingestion_report_path, report)

    logger.info("Ingestion complete.")
    logger.info(json.dumps(report, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1 ingestion pipeline for KazNU Agentic RAG system."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to settings YAML file.",
    )
    args = parser.parse_args()

    run_ingestion(Path(args.config))


if __name__ == "__main__":
    main()