import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple

import fitz  # PyMuPDF
import pdfplumber
import pandas as pd

from kaznu_rag.schemas import DocumentRecord
from kaznu_rag.preprocess.text_cleaning import clean_text
from kaznu_rag.utils import sha256_text

logger = logging.getLogger(__name__)


def infer_document_group(pdf_path: Path) -> str:
    name = pdf_path.name.lower()

    if "academic-policy" in name or "academic" in name:
        return "academic_policy"
    if "ai_regulation" in name or "regulation" in name:
        return "ai_regulation"
    if "tuition" in name:
        return "tuition_fee"
    if "booklet" in name:
        return "university_booklet"

    return "unknown_pdf"


def infer_authority_level(document_group: str) -> str:
    if document_group in {"academic_policy", "ai_regulation", "tuition_fee"}:
        return "official_policy_or_regulation"
    if document_group == "university_booklet":
        return "official_brochure"
    return "unknown"


def extract_pdf_text_records(
    pdf_path: Path,
    min_text_chars: int = 100,
) -> List[DocumentRecord]:
    records: List[DocumentRecord] = []
    document_group = infer_document_group(pdf_path)
    authority_level = infer_authority_level(document_group)
    collected_at = datetime.now(timezone.utc).isoformat()

    logger.info("Extracting PDF text: %s", pdf_path)

    doc = fitz.open(pdf_path)

    for page_index, page in enumerate(doc, start=1):
        raw_text = page.get_text("text")
        text = clean_text(raw_text)

        if len(text) < min_text_chars:
            logger.warning(
                "Skipping short PDF page: file=%s page=%s chars=%s",
                pdf_path.name,
                page_index,
                len(text),
            )
            continue

        checksum = sha256_text(text)
        doc_id = f"pdf::{pdf_path.stem}::page-{page_index}::{checksum[:12]}"

        records.append(
            DocumentRecord(
                doc_id=doc_id,
                source_type="pdf",
                source_name=pdf_path.name,
                content_type="text",
                text=text,
                page_number=page_index,
                title=pdf_path.stem,
                metadata={
                    "document_group": document_group,
                    "authority_level": authority_level,
                    "file_path": str(pdf_path),
                    "collected_at": collected_at,
                    "checksum": checksum,
                    "token_estimate": max(1, len(text.split())),
                },
            )
        )

    doc.close()
    return records


def clean_table_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\n", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

    df = df.replace({"None": "", "nan": "", "NaN": ""})
    return df


def dataframe_to_plain_text(df: pd.DataFrame) -> str:
    return " ".join(
        str(value).strip()
        for value in df.fillna("").values.flatten().tolist()
        if str(value).strip()
    )


def table_to_markdown(df: pd.DataFrame) -> str:
    df = df.fillna("")
    return df.to_markdown(index=False)


def is_probably_noise_table(
    df: pd.DataFrame,
    pdf_path: Path,
    page_index: int,
) -> bool:
    document_group = infer_document_group(pdf_path)
    rows, cols = df.shape
    plain_text = dataframe_to_plain_text(df).lower()

    if not plain_text:
        return True

    # Tuition data is critical; keep all tuition tables for later normalization.
    if document_group == "tuition_fee":
        return False

    boilerplate_terms = [
        "date:",
        "edition:",
        "page",
        "kazakh national university",
        "academic policy",
        "п казну",
        "нао",
    ]

    boilerplate_hits = sum(1 for term in boilerplate_terms if term in plain_text)

    # Common false table: repeated PDF header/footer layout.
    if rows <= 2 and cols <= 4 and boilerplate_hits >= 2:
        return True

    # Very small non-tuition tables usually add retrieval noise.
    if rows <= 2 and len(plain_text) < 500:
        return True

    # Single-column contents tables are usually duplicated in normal text extraction.
    if cols == 1 and rows <= 10 and document_group != "tuition_fee":
        return True

    return False


def extract_pdf_table_records(
    pdf_path: Path,
    output_table_dir: Path,
) -> Tuple[List[DocumentRecord], List[dict]]:
    records: List[DocumentRecord] = []
    table_manifest: List[dict] = []

    output_table_dir.mkdir(parents=True, exist_ok=True)

    document_group = infer_document_group(pdf_path)
    authority_level = infer_authority_level(document_group)
    collected_at = datetime.now(timezone.utc).isoformat()

    logger.info("Extracting PDF tables: %s", pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables() or []

            for table_index, table in enumerate(tables, start=1):
                if not table or len(table) < 2:
                    continue

                raw_df = pd.DataFrame(table)
                df = clean_table_dataframe(raw_df)

                if df.empty:
                    continue

                if is_probably_noise_table(df, pdf_path, page_index):
                    logger.info(
                        "Skipping noise table: file=%s page=%s table=%s rows=%s cols=%s",
                        pdf_path.name,
                        page_index,
                        table_index,
                        df.shape[0],
                        df.shape[1],
                    )
                    continue

                table_id = f"{pdf_path.stem}_p{page_index}_t{table_index}"
                csv_path = output_table_dir / f"{table_id}.csv"
                df.to_csv(csv_path, index=False, encoding="utf-8")

                markdown_text = table_to_markdown(df)
                checksum = sha256_text(markdown_text)
                doc_id = f"pdf-table::{table_id}::{checksum[:12]}"

                records.append(
                    DocumentRecord(
                        doc_id=doc_id,
                        source_type="pdf",
                        source_name=pdf_path.name,
                        content_type="table",
                        text=markdown_text,
                        page_number=page_index,
                        title=f"{pdf_path.stem} table {table_index}",
                        metadata={
                            "document_group": document_group,
                            "authority_level": authority_level,
                            "file_path": str(pdf_path),
                            "table_id": table_id,
                            "csv_path": str(csv_path),
                            "collected_at": collected_at,
                            "checksum": checksum,
                            "token_estimate": max(1, len(markdown_text.split())),
                        },
                    )
                )

                table_manifest.append(
                    {
                        "table_id": table_id,
                        "source_name": pdf_path.name,
                        "page_number": page_index,
                        "csv_path": str(csv_path),
                        "rows": int(df.shape[0]),
                        "columns": int(df.shape[1]),
                    }
                )

    return records, table_manifest