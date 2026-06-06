#!/usr/bin/env bash

set -e

# ============================================================
# Thesis Appendices Evidence Package Builder
# Project: Agentic-KaZNU-assistant
#
# Purpose:
# Create a clean appendices evidence folder and ZIP file for:
# - Appendix A: Dataset and knowledge sources
# - Appendix B: Configuration and environment
# - Appendix C: Evaluation question sets
# - Appendix D: Evaluation rubric/prompts
# - Appendix E: Sample Baseline and Agentic outputs
# - Appendix F: Agentic trace examples
# - Appendix G: Human evaluation template
# - Appendix H: Selected code snippets
# - Appendix I: Additional figures and reports
# ============================================================

PACKAGE_DIR="thesis_appendices_evidence"
ZIP_NAME="thesis_appendices_evidence.zip"

echo "============================================================"
echo "Preparing thesis appendices evidence package"
echo "Package folder: $PACKAGE_DIR"
echo "ZIP file: $ZIP_NAME"
echo "============================================================"

# ------------------------------------------------------------
# Clean old package
# ------------------------------------------------------------
rm -rf "$PACKAGE_DIR"
rm -f "$ZIP_NAME"

# ------------------------------------------------------------
# Create appendix folder structure
# ------------------------------------------------------------
mkdir -p "$PACKAGE_DIR/Appendix_A_Dataset_and_Knowledge_Sources"
mkdir -p "$PACKAGE_DIR/Appendix_B_Configuration_and_Environment"
mkdir -p "$PACKAGE_DIR/Appendix_C_Evaluation_Question_Sets"
mkdir -p "$PACKAGE_DIR/Appendix_D_Evaluation_Rubric_and_Prompts"
mkdir -p "$PACKAGE_DIR/Appendix_E_Sample_System_Outputs"
mkdir -p "$PACKAGE_DIR/Appendix_F_Agentic_Trace_Examples"
mkdir -p "$PACKAGE_DIR/Appendix_G_Human_Evaluation_Template"
mkdir -p "$PACKAGE_DIR/Appendix_H_Selected_Code_Snippets"
mkdir -p "$PACKAGE_DIR/Appendix_I_Additional_Figures_and_Reports"
mkdir -p "$PACKAGE_DIR/notes"

# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
copy_file_if_exists() {
    SRC="$1"
    DEST="$2"

    if [ -f "$SRC" ]; then
        mkdir -p "$(dirname "$DEST")"
        cp "$SRC" "$DEST"
        echo "Copied file: $SRC"
    else
        echo "Skipped missing file: $SRC"
    fi
}

copy_dir_if_exists() {
    SRC="$1"
    DEST="$2"

    if [ -d "$SRC" ]; then
        mkdir -p "$(dirname "$DEST")"
        cp -r "$SRC" "$DEST"
        echo "Copied folder: $SRC"
    else
        echo "Skipped missing folder: $SRC"
    fi
}

copy_matching_files() {
    PATTERN="$1"
    DEST_DIR="$2"

    mkdir -p "$DEST_DIR"

    FOUND=0
    for FILE in $PATTERN; do
        if [ -f "$FILE" ]; then
            cp "$FILE" "$DEST_DIR/"
            echo "Copied matched file: $FILE"
            FOUND=1
        fi
    done

    if [ "$FOUND" -eq 0 ]; then
        echo "No files matched: $PATTERN"
    fi
}

# ============================================================
# APPENDIX A — DATASET AND KNOWLEDGE SOURCES
# ============================================================
echo ""
echo "Appendix A: Dataset and knowledge sources"

copy_file_if_exists "data/raw/urls.txt" "$PACKAGE_DIR/Appendix_A_Dataset_and_Knowledge_Sources/urls.txt"
copy_file_if_exists "data/processed/ingestion_report.json" "$PACKAGE_DIR/Appendix_A_Dataset_and_Knowledge_Sources/ingestion_report.json"
copy_file_if_exists "data/processed/chunk_report.json" "$PACKAGE_DIR/Appendix_A_Dataset_and_Knowledge_Sources/chunk_report.json"
copy_file_if_exists "data/processed/tuition_fees_normalized.jsonl" "$PACKAGE_DIR/Appendix_A_Dataset_and_Knowledge_Sources/tuition_fees_normalized.jsonl"

cat > "$PACKAGE_DIR/Appendix_A_Dataset_and_Knowledge_Sources/PDF_SOURCE_LIST.md" <<'EOF'
# PDF Source List

The practical implementation used selected university-related PDF sources, including:

1. Academic Policy 2025–2026
2. KazNU Booklet 2025
3. Tuition Fee 2022–2023
4. AI Regulation 2024

These sources were used together with official university web URLs and structured tuition-fee facts.
EOF

# ============================================================
# APPENDIX B — CONFIGURATION AND ENVIRONMENT
# ============================================================
echo ""
echo "Appendix B: Configuration and environment"

copy_file_if_exists "config/settings.yaml" "$PACKAGE_DIR/Appendix_B_Configuration_and_Environment/settings.yaml"
copy_file_if_exists "requirements.txt" "$PACKAGE_DIR/Appendix_B_Configuration_and_Environment/requirements.txt"
copy_file_if_exists "pyproject.toml" "$PACKAGE_DIR/Appendix_B_Configuration_and_Environment/pyproject.toml"
copy_file_if_exists ".env.example" "$PACKAGE_DIR/Appendix_B_Configuration_and_Environment/env_example.txt"
copy_file_if_exists "README.md" "$PACKAGE_DIR/Appendix_B_Configuration_and_Environment/README.md"

cat > "$PACKAGE_DIR/Appendix_B_Configuration_and_Environment/ENVIRONMENT_NOTE.md" <<'EOF'
# Environment Note

The real `.env` file is intentionally excluded because it contains API keys and private credentials.

Only `.env.example` is included as a safe configuration template.
EOF

# ============================================================
# APPENDIX C — EVALUATION QUESTION SETS
# ============================================================
echo ""
echo "Appendix C: Evaluation question sets"

copy_file_if_exists "evaluation/baseline_test_questions.json" "$PACKAGE_DIR/Appendix_C_Evaluation_Question_Sets/baseline_test_questions.json"
copy_file_if_exists "evaluation/complex_questions_30.json" "$PACKAGE_DIR/Appendix_C_Evaluation_Question_Sets/complex_questions_30.json"
copy_file_if_exists "evaluation/agentic_complex_questions.json" "$PACKAGE_DIR/Appendix_C_Evaluation_Question_Sets/agentic_complex_questions.json"

# ============================================================
# APPENDIX D — EVALUATION RUBRIC AND PROMPTS
# ============================================================
echo ""
echo "Appendix D: Evaluation rubric and prompts"

copy_file_if_exists "src/kaznu_rag/evaluation/evaluate_baseline.py" "$PACKAGE_DIR/Appendix_D_Evaluation_Rubric_and_Prompts/evaluate_baseline.py"
copy_file_if_exists "src/kaznu_rag/evaluation/compare_results.py" "$PACKAGE_DIR/Appendix_D_Evaluation_Rubric_and_Prompts/compare_results.py"
copy_file_if_exists "src/kaznu_rag/evaluation/create_human_eval_template.py" "$PACKAGE_DIR/Appendix_D_Evaluation_Rubric_and_Prompts/create_human_eval_template.py"

cat > "$PACKAGE_DIR/Appendix_D_Evaluation_Rubric_and_Prompts/EVALUATION_METRICS.md" <<'EOF'
# Evaluation Metrics Used in the Thesis

The evaluation framework used the following dimensions:

1. Faithfulness score
2. Answer relevance score
3. Context relevance score
4. Completeness score
5. Citation quality score
6. Hallucination score
7. Hallucination detected
8. Hallucination rate
9. Latency
10. Agentic trace metrics:
   - number of subqueries
   - validated sources
   - rejected sources

These metrics support the comparison between Baseline RAG and Agentic RAG.
EOF

# ============================================================
# APPENDIX E — SAMPLE SYSTEM OUTPUTS
# ============================================================
echo ""
echo "Appendix E: Sample Baseline and Agentic system outputs"

copy_file_if_exists "outputs/baseline_rag/baseline_complex30_results.jsonl" "$PACKAGE_DIR/Appendix_E_Sample_System_Outputs/baseline_complex30_results.jsonl"
copy_file_if_exists "outputs/agentic_rag/agentic_complex30_results.jsonl" "$PACKAGE_DIR/Appendix_E_Sample_System_Outputs/agentic_complex30_results.jsonl"

copy_matching_files "outputs/baseline_rag/*test*.jsonl" "$PACKAGE_DIR/Appendix_E_Sample_System_Outputs/test_outputs"
copy_matching_files "outputs/agentic_rag/*test*.jsonl" "$PACKAGE_DIR/Appendix_E_Sample_System_Outputs/test_outputs"

# Create smaller sample files if Python is available
python - <<'PY'
import json
from pathlib import Path

samples = [
    (
        Path("outputs/baseline_rag/baseline_complex30_results.jsonl"),
        Path("thesis_appendices_evidence/Appendix_E_Sample_System_Outputs/baseline_complex30_sample_5.jsonl"),
        5,
    ),
    (
        Path("outputs/agentic_rag/agentic_complex30_results.jsonl"),
        Path("thesis_appendices_evidence/Appendix_E_Sample_System_Outputs/agentic_complex30_sample_5.jsonl"),
        5,
    ),
]

for src, dst, limit in samples:
    if not src.exists():
        print(f"Skipped sample creation, missing: {src}")
        continue

    dst.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if line.strip():
                f_out.write(line)
                count += 1
            if count >= limit:
                break

    print(f"Created sample file: {dst} ({count} records)")
PY

# ============================================================
# APPENDIX F — AGENTIC TRACE EXAMPLES
# ============================================================
echo ""
echo "Appendix F: Agentic trace examples"

copy_file_if_exists "src/kaznu_rag/agentic/query_decomposition.py" "$PACKAGE_DIR/Appendix_F_Agentic_Trace_Examples/query_decomposition.py"
copy_file_if_exists "src/kaznu_rag/agentic/source_validator.py" "$PACKAGE_DIR/Appendix_F_Agentic_Trace_Examples/source_validator.py"
copy_file_if_exists "src/kaznu_rag/agentic/source_sufficiency.py" "$PACKAGE_DIR/Appendix_F_Agentic_Trace_Examples/source_sufficiency.py"
copy_file_if_exists "src/kaznu_rag/agentic/reflection_agent.py" "$PACKAGE_DIR/Appendix_F_Agentic_Trace_Examples/reflection_agent.py"
copy_file_if_exists "src/kaznu_rag/agentic/adaptive_rag.py" "$PACKAGE_DIR/Appendix_F_Agentic_Trace_Examples/adaptive_rag.py"

copy_file_if_exists "outputs/agentic_rag/agentic_complex30_results.jsonl" "$PACKAGE_DIR/Appendix_F_Agentic_Trace_Examples/agentic_complex30_results.jsonl"

# Extract one trace-like example if possible
python - <<'PY'
import json
from pathlib import Path

src = Path("outputs/agentic_rag/agentic_complex30_results.jsonl")
dst = Path("thesis_appendices_evidence/Appendix_F_Agentic_Trace_Examples/agentic_trace_example.json")

if not src.exists():
    print(f"Skipped trace example creation, missing: {src}")
else:
    chosen = None

    with src.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            question = (row.get("question") or "").lower()

            if any(term in question for term in ["total cost", "insurance", "visa", "medical", "hiv", "generative ai", "ai tools"]):
                chosen = row
                break

            if chosen is None:
                chosen = row

    if chosen:
        with dst.open("w", encoding="utf-8") as f:
            json.dump(chosen, f, ensure_ascii=False, indent=2)
        print(f"Created trace example: {dst}")
PY

cat > "$PACKAGE_DIR/Appendix_F_Agentic_Trace_Examples/TRACE_NOTE.md" <<'EOF'
# Agentic Trace Note

This appendix supports the transparency contribution of Agentic RAG.

Important trace fields may include:
- original question
- generated subqueries
- retrieved context
- validated context
- rejected context
- source sufficiency result
- reflection result
- final answer
- latency
EOF

# ============================================================
# APPENDIX G — HUMAN EVALUATION TEMPLATE
# ============================================================
echo ""
echo "Appendix G: Human evaluation template"

copy_file_if_exists "outputs/evaluation/human_eval/human_evaluation_template.csv" "$PACKAGE_DIR/Appendix_G_Human_Evaluation_Template/human_evaluation_template.csv"

cat > "$PACKAGE_DIR/Appendix_G_Human_Evaluation_Template/HUMAN_EVALUATION_NOTE.md" <<'EOF'
# Human Evaluation Template Note

The human evaluation template was prepared to support future or pilot reviewer-based assessment.

Reviewer dimensions:
1. Correctness
2. Completeness
3. Clarity
4. Usefulness
5. Trustworthiness
6. Source transparency
7. Observed hallucination
8. Reviewer comments

If no completed human evaluation was conducted, this appendix should be described as a prepared evaluation instrument, not as completed human-evaluation results.
EOF

# ============================================================
# APPENDIX H — SELECTED CODE SNIPPETS
# ============================================================
echo ""
echo "Appendix H: Selected code snippets"

copy_file_if_exists "src/kaznu_rag/chunking/chunk_documents.py" "$PACKAGE_DIR/Appendix_H_Selected_Code_Snippets/chunk_documents.py"
copy_file_if_exists "src/kaznu_rag/chunking/build_vectorstore.py" "$PACKAGE_DIR/Appendix_H_Selected_Code_Snippets/build_vectorstore.py"

copy_file_if_exists "src/kaznu_rag/rag/retriever.py" "$PACKAGE_DIR/Appendix_H_Selected_Code_Snippets/retriever.py"
copy_file_if_exists "src/kaznu_rag/rag/prompt_templates.py" "$PACKAGE_DIR/Appendix_H_Selected_Code_Snippets/prompt_templates.py"
copy_file_if_exists "src/kaznu_rag/rag/qa_chain.py" "$PACKAGE_DIR/Appendix_H_Selected_Code_Snippets/qa_chain.py"
copy_file_if_exists "src/kaznu_rag/rag/baseline_rag.py" "$PACKAGE_DIR/Appendix_H_Selected_Code_Snippets/baseline_rag.py"
copy_file_if_exists "src/kaznu_rag/rag/llm_client.py" "$PACKAGE_DIR/Appendix_H_Selected_Code_Snippets/llm_client.py"

copy_file_if_exists "src/kaznu_rag/agentic/query_decomposition.py" "$PACKAGE_DIR/Appendix_H_Selected_Code_Snippets/query_decomposition.py"
copy_file_if_exists "src/kaznu_rag/agentic/multi_query_retriever.py" "$PACKAGE_DIR/Appendix_H_Selected_Code_Snippets/multi_query_retriever.py"
copy_file_if_exists "src/kaznu_rag/agentic/source_validator.py" "$PACKAGE_DIR/Appendix_H_Selected_Code_Snippets/source_validator.py"
copy_file_if_exists "src/kaznu_rag/agentic/source_sufficiency.py" "$PACKAGE_DIR/Appendix_H_Selected_Code_Snippets/source_sufficiency.py"
copy_file_if_exists "src/kaznu_rag/agentic/reflection_agent.py" "$PACKAGE_DIR/Appendix_H_Selected_Code_Snippets/reflection_agent.py"
copy_file_if_exists "src/kaznu_rag/agentic/adaptive_rag.py" "$PACKAGE_DIR/Appendix_H_Selected_Code_Snippets/adaptive_rag.py"

cat > "$PACKAGE_DIR/Appendix_H_Selected_Code_Snippets/CODE_SNIPPET_NOTE.md" <<'EOF'
# Selected Code Snippets Note

This appendix contains selected implementation files relevant to:
- chunking
- vector-store construction
- Baseline RAG
- Agentic RAG
- source validation
- source sufficiency scoring
- reflection
- adaptive routing

The full project repository should be used for complete reproducibility.
EOF

# ============================================================
# APPENDIX I — ADDITIONAL FIGURES AND REPORTS
# ============================================================
echo ""
echo "Appendix I: Additional figures and reports"

copy_file_if_exists "outputs/evaluation/comparison_complex30/rag_comparison_report.md" "$PACKAGE_DIR/Appendix_I_Additional_Figures_and_Reports/rag_comparison_report_complex30.md"
copy_file_if_exists "outputs/evaluation/comparison/rag_comparison_report.md" "$PACKAGE_DIR/Appendix_I_Additional_Figures_and_Reports/rag_comparison_report.md"

copy_matching_files "outputs/baseline_rag/*summary*.json" "$PACKAGE_DIR/Appendix_I_Additional_Figures_and_Reports/summaries/baseline"
copy_matching_files "outputs/agentic_rag/*summary*.json" "$PACKAGE_DIR/Appendix_I_Additional_Figures_and_Reports/summaries/agentic"
copy_matching_files "outputs/evaluation/*summary*.json" "$PACKAGE_DIR/Appendix_I_Additional_Figures_and_Reports/summaries/evaluation"
copy_matching_files "outputs/evaluation/comparison*/*.json" "$PACKAGE_DIR/Appendix_I_Additional_Figures_and_Reports/summaries/comparison"

copy_dir_if_exists "outputs/thesis_figures" "$PACKAGE_DIR/Appendix_I_Additional_Figures_and_Reports/thesis_figures"
copy_dir_if_exists "outputs/thesis_figures_polished" "$PACKAGE_DIR/Appendix_I_Additional_Figures_and_Reports/thesis_figures_polished"
copy_dir_if_exists "outputs/evaluation/plots_complex30" "$PACKAGE_DIR/Appendix_I_Additional_Figures_and_Reports/plots_complex30"
copy_dir_if_exists "outputs/evaluation/plots" "$PACKAGE_DIR/Appendix_I_Additional_Figures_and_Reports/plots"

# ============================================================
# CREATE APPENDIX GUIDE
# ============================================================
echo ""
echo "Creating appendix guide"

cat > "$PACKAGE_DIR/APPENDIX_GUIDE.md" <<'EOF'
# Thesis Appendices Evidence Guide

This folder contains selected evidence for the appendices of the MSc thesis:

## Appendix A — Dataset and Knowledge Sources
Includes URL list, ingestion report, chunk report, and tuition-fee normalized data.

## Appendix B — Configuration and Environment
Includes settings, requirements, pyproject, README, and safe `.env.example`.

## Appendix C — Evaluation Question Sets
Includes Baseline RAG and complex-query evaluation question sets.

## Appendix D — Evaluation Rubric and Prompts
Includes evaluation scripts and metric description.

## Appendix E — Sample System Outputs
Includes sample Baseline RAG and Agentic RAG output files.

## Appendix F — Agentic Trace Examples
Includes trace-related files and a selected trace example if available.

## Appendix G — Human Evaluation Template
Includes human evaluation template and note.

## Appendix H — Selected Code Snippets
Includes selected implementation files for reproducibility.

## Appendix I — Additional Figures and Reports
Includes comparison reports, summary JSON files, and generated figures.

Important:
- The real `.env` file is not included.
- Large vector-store files are not included.
- This package is for thesis appendices and supervisor review, not for production deployment.
EOF

# ============================================================
# SAFETY CHECK
# ============================================================
echo ""
echo "Running safety check..."

if find "$PACKAGE_DIR" -name ".env" -o -name ".env.*" | grep -q .; then
    echo "WARNING: An .env file was found in the package. Remove it before sharing."
else
    echo "Safety check passed: no .env file found."
fi

# ============================================================
# CREATE ZIP
# ============================================================
echo ""
echo "Creating ZIP file..."

if command -v zip >/dev/null 2>&1; then
    zip -r "$ZIP_NAME" "$PACKAGE_DIR" >/dev/null
    echo "ZIP created using zip: $ZIP_NAME"
else
    echo "zip command not found. Trying PowerShell Compress-Archive..."
    powershell.exe -NoProfile -Command "Compress-Archive -Path '$PACKAGE_DIR' -DestinationPath '$ZIP_NAME' -Force"

    if [ -f "$ZIP_NAME" ]; then
        echo "ZIP created using PowerShell: $ZIP_NAME"
    else
        echo "Could not create ZIP automatically."
        echo "Please right-click the folder and choose: Send to -> Compressed zipped folder"
    fi
fi

# ============================================================
# FINAL SUMMARY
# ============================================================
echo ""
echo "============================================================"
echo "Appendices evidence package created successfully."
echo "Folder: $PACKAGE_DIR"
echo "ZIP: $ZIP_NAME"
echo "============================================================"

echo ""
echo "Package size:"
du -sh "$PACKAGE_DIR" 2>/dev/null || true

if [ -f "$ZIP_NAME" ]; then
    du -sh "$ZIP_NAME" 2>/dev/null || true
fi

echo ""
echo "Done."