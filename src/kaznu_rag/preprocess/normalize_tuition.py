import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


COLUMN_MAPPING = {
    2: ("cis", "bachelor", "kazakh"),
    3: ("cis", "bachelor", "russian"),
    4: ("cis", "bachelor", "english"),
    5: ("cis", "masters", "kazakh"),
    6: ("cis", "masters", "russian"),
    7: ("cis", "masters", "english"),
    8: ("cis", "phd", "kazakh"),
    9: ("cis", "phd", "russian"),
    10: ("cis", "phd", "english"),
    11: ("far_abroad", "bachelor", "kazakh"),
    12: ("far_abroad", "bachelor", "russian"),
    13: ("far_abroad", "bachelor", "english"),
    14: ("far_abroad", "masters", "kazakh"),
    15: ("far_abroad", "masters", "russian"),
    16: ("far_abroad", "masters", "english"),
    17: ("far_abroad", "phd", "kazakh"),
    18: ("far_abroad", "phd", "russian"),
    19: ("far_abroad", "phd", "english"),
}


def clean_fee(value) -> Optional[int]:
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in {"-", "", "nan", "NaN", "None"}:
        return None

    value = value.replace(" ", "").replace(",", "")

    try:
        fee = int(float(value))
    except ValueError:
        return None

    # Prevent header values like 2, 3, 4 from becoming tuition fees.
    if fee < 100_000:
        return None

    return fee


def normalize_faculty_name(value) -> str:
    if pd.isna(value):
        return ""

    return " ".join(str(value).replace("\n", " ").split()).strip()


def parse_faculty_index(value) -> Optional[int]:
    if pd.isna(value):
        return None

    try:
        idx = int(float(str(value).strip()))
    except ValueError:
        return None

    if 1 <= idx <= 16:
        return idx

    return None


def is_valid_faculty_row(row: pd.Series) -> bool:
    if len(row) < 20:
        return False

    faculty_index = parse_faculty_index(row.iloc[0])
    faculty = normalize_faculty_name(row.iloc[1])

    if faculty_index is None:
        return False

    if not faculty:
        return False

    invalid_faculty_values = {
        "faculties",
        "faculty",
        "nan",
        "none",
        "1",
        "2",
        "3",
        "4",
        "5",
    }

    if faculty.lower() in invalid_faculty_values:
        return False

    return True


def load_tuition_tables(table_dir: Path) -> List[Path]:
    return sorted(table_dir.glob("TUITION_FEE_2022-2023*.csv"))


def normalize_tuition_tables(
    table_dir: Path,
    output_path: Path,
) -> List[Dict]:
    normalized_records: List[Dict] = []

    csv_files = load_tuition_tables(table_dir)

    logger.info("Found tuition CSV files: %s", len(csv_files))

    for csv_path in csv_files:
        logger.info("Processing tuition CSV: %s", csv_path)

        df = pd.read_csv(csv_path, header=None)

        if df.shape[1] < 20:
            logger.info(
                "Skipping non-main tuition table: %s | shape=%s",
                csv_path.name,
                df.shape,
            )
            continue

        for _, row in df.iterrows():
            if not is_valid_faculty_row(row):
                continue

            faculty_index = parse_faculty_index(row.iloc[0])
            faculty = normalize_faculty_name(row.iloc[1])

            for col_index, mapping in COLUMN_MAPPING.items():
                applicant_region, degree_level, language = mapping
                fee = clean_fee(row.iloc[col_index])

                if fee is None:
                    continue

                normalized_records.append(
                    {
                        "faculty_index": faculty_index,
                        "faculty": faculty,
                        "applicant_region": applicant_region,
                        "degree_level": degree_level,
                        "language": language,
                        "tuition_fee_kzt": fee,
                        "currency": "KZT",
                        "academic_year": "2022-2023",
                        "source_file": csv_path.name,
                        "source_type": "pdf_table",
                        "source_name": "TUITION_FEE_2022-2023.pdf",
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for record in normalized_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Normalized tuition records saved: %s", output_path)
    logger.info("Total normalized tuition rows: %s", len(normalized_records))

    return normalized_records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    records = normalize_tuition_tables(
        table_dir=Path("data/interim/pdf_tables"),
        output_path=Path("data/processed/tuition_fees_normalized.jsonl"),
    )

    print(f"Normalized rows: {len(records)}")