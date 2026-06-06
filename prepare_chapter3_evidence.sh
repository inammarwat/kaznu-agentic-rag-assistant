#!/usr/bin/env bash

set -e

# ============================================================
# Chapter 3 Evidence Package Builder
# Project: Agentic-KaZNU-assistant
# Purpose: Collect implementation, experiment, and result files
#          for thesis Chapter 3: Implementation, Experiments,
#          and Results.
# ============================================================

PACKAGE_DIR="chapter3_implementation_results_evidence"
ZIP_NAME="chapter3_implementation_results_evidence.zip"

echo "============================================================"
echo "Preparing Chapter 3 evidence package"
echo "Output folder: $PACKAGE_DIR"
echo "ZIP file: $ZIP_NAME"
echo "============================================================"

# ------------------------------------------------------------
# Clean old package
# ------------------------------------------------------------
if [ -d "$PACKAGE_DIR" ]; then
    echo "Removing old package folder..."
    rm -rf "$PACKAGE_DIR"
fi

if [ -f "$ZIP_NAME" ]; then
    echo "Removing old ZIP file..."
    rm -f "$ZIP_NAME"
fi

# ------------------------------------------------------------
# Create folder structure
# ------------------------------------------------------------
mkdir -p "$PACKAGE_DIR/config"
mkdir -p "$PACKAGE_DIR/src/kaznu_rag"
mkdir -p "$PACKAGE_DIR/reports"
mkdir -p "$PACKAGE_DIR/evaluation_questions"
mkdir -p "$PACKAGE_DIR/evaluation_summaries"
mkdir -p "$PACKAGE_DIR/sample_outputs"
mkdir -p "$PACKAGE_DIR/figures"
mkdir -p "$PACKAGE_DIR/human_evaluation"
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
        echo "No files matched pattern: $PATTERN"
    fi
}

# ------------------------------------------------------------
# 1. Project overview and reproducibility files
# ------------------------------------------------------------
echo ""
echo "1. Copying project overview files..."

copy_file_if_exists "README.md" "$PACKAGE_DIR/README.md"
copy_file_if_exists "requirements.txt" "$PACKAGE_DIR/requirements.txt"
copy_file_if_exists "pyproject.toml" "$PACKAGE_DIR/pyproject.toml"
copy_file_if_exists "config/settings.yaml" "$PACKAGE_DIR/config/settings.yaml"

# ------------------------------------------------------------
# 2. Main source code files/folders
# ------------------------------------------------------------
echo ""
echo "2. Copying source code folders..."

copy_file_if_exists "src/kaznu_rag/pipeline_ingest.py" "$PACKAGE_DIR/src/kaznu_rag/pipeline_ingest.py"
copy_file_if_exists "src/kaznu_rag/schemas.py" "$PACKAGE_DIR/src/kaznu_rag/schemas.py"
copy_file_if_exists "src/kaznu_rag/utils.py" "$PACKAGE_DIR/src/kaznu_rag/utils.py"

copy_dir_if_exists "src/kaznu_rag/ingest" "$PACKAGE_DIR/src/kaznu_rag/ingest"
copy_dir_if_exists "src/kaznu_rag/preprocess" "$PACKAGE_DIR/src/kaznu_rag/preprocess"
copy_dir_if_exists "src/kaznu_rag/chunking" "$PACKAGE_DIR/src/kaznu_rag/chunking"
copy_dir_if_exists "src/kaznu_rag/rag" "$PACKAGE_DIR/src/kaznu_rag/rag"
copy_dir_if_exists "src/kaznu_rag/agentic" "$PACKAGE_DIR/src/kaznu_rag/agentic"
copy_dir_if_exists "src/kaznu_rag/evaluation" "$PACKAGE_DIR/src/kaznu_rag/evaluation"

# ------------------------------------------------------------
# 3. Processed reports
# ------------------------------------------------------------
echo ""
echo "3. Copying processed reports..."

copy_file_if_exists "data/processed/ingestion_report.json" "$PACKAGE_DIR/reports/ingestion_report.json"
copy_file_if_exists "data/processed/chunk_report.json" "$PACKAGE_DIR/reports/chunk_report.json"
copy_file_if_exists "data/processed/vectorstore_report.json" "$PACKAGE_DIR/reports/vectorstore_report.json"

# Some projects may save vectorstore report elsewhere
copy_file_if_exists "data/vectorstore/vectorstore_report.json" "$PACKAGE_DIR/reports/vectorstore_report_alt.json"
copy_file_if_exists "outputs/vectorstore_report.json" "$PACKAGE_DIR/reports/vectorstore_report_outputs.json"

# ------------------------------------------------------------
# 4. Evaluation question files
# ------------------------------------------------------------
echo ""
echo "4. Copying evaluation question files..."

copy_file_if_exists "evaluation/baseline_test_questions.json" "$PACKAGE_DIR/evaluation_questions/baseline_test_questions.json"
copy_file_if_exists "evaluation/complex_questions_30.json" "$PACKAGE_DIR/evaluation_questions/complex_questions_30.json"
copy_file_if_exists "evaluation/agentic_complex_questions.json" "$PACKAGE_DIR/evaluation_questions/agentic_complex_questions.json"

# ------------------------------------------------------------
# 5. Batch summaries and evaluation summaries
# ------------------------------------------------------------
echo ""
echo "5. Copying summary JSON files..."

copy_matching_files "outputs/baseline_rag/*summary*.json" "$PACKAGE_DIR/evaluation_summaries/baseline_rag"
copy_matching_files "outputs/agentic_rag/*summary*.json" "$PACKAGE_DIR/evaluation_summaries/agentic_rag"
copy_matching_files "outputs/evaluation/*summary*.json" "$PACKAGE_DIR/evaluation_summaries/evaluation"

# Copy nested comparison summaries/reports if present
copy_matching_files "outputs/evaluation/comparison*/*.json" "$PACKAGE_DIR/evaluation_summaries/comparison"
copy_matching_files "outputs/evaluation/comparison*/*.md" "$PACKAGE_DIR/reports"

# ------------------------------------------------------------
# 6. Main comparison report
# ------------------------------------------------------------
echo ""
echo "6. Copying comparison reports..."

copy_file_if_exists "outputs/evaluation/comparison_complex30/rag_comparison_report.md" "$PACKAGE_DIR/reports/rag_comparison_report_complex30.md"
copy_file_if_exists "outputs/evaluation/comparison/rag_comparison_report.md" "$PACKAGE_DIR/reports/rag_comparison_report.md"

# ------------------------------------------------------------
# 7. Sample outputs
# ------------------------------------------------------------
echo ""
echo "7. Copying sample JSONL output files..."

# These can be large, so keep exact files if present.
# If they are too large, you can delete them from the package later.
copy_file_if_exists "outputs/baseline_rag/baseline_complex30_results.jsonl" "$PACKAGE_DIR/sample_outputs/baseline_complex30_results.jsonl"
copy_file_if_exists "outputs/agentic_rag/agentic_complex30_results.jsonl" "$PACKAGE_DIR/sample_outputs/agentic_complex30_results.jsonl"
copy_file_if_exists "outputs/evaluation/baseline_complex30_eval_results.jsonl" "$PACKAGE_DIR/sample_outputs/baseline_complex30_eval_results.jsonl"
copy_file_if_exists "outputs/evaluation/agentic_complex30_eval_results.jsonl" "$PACKAGE_DIR/sample_outputs/agentic_complex30_eval_results.jsonl"

# Test outputs if present
copy_matching_files "outputs/baseline_rag/*test*.jsonl" "$PACKAGE_DIR/sample_outputs"
copy_matching_files "outputs/agentic_rag/*test*.jsonl" "$PACKAGE_DIR/sample_outputs"
copy_matching_files "outputs/evaluation/*test*.jsonl" "$PACKAGE_DIR/sample_outputs"

# ------------------------------------------------------------
# 8. Figures and graphs
# ------------------------------------------------------------
echo ""
echo "8. Copying figures..."

copy_dir_if_exists "outputs/thesis_figures" "$PACKAGE_DIR/figures/thesis_figures"
copy_dir_if_exists "outputs/thesis_figures_polished" "$PACKAGE_DIR/figures/thesis_figures_polished"

copy_dir_if_exists "outputs/evaluation/plots_complex30" "$PACKAGE_DIR/figures/plots_complex30"
copy_dir_if_exists "outputs/evaluation/plots" "$PACKAGE_DIR/figures/plots"

# If figures were generated in root or another folder
copy_matching_files "*.png" "$PACKAGE_DIR/figures/root_png"
copy_matching_files "*.svg" "$PACKAGE_DIR/figures/root_svg"
copy_matching_files "*.pdf" "$PACKAGE_DIR/figures/root_pdf"

# ------------------------------------------------------------
# 9. Human evaluation template
# ------------------------------------------------------------
echo ""
echo "9. Copying human evaluation files..."

copy_file_if_exists "outputs/evaluation/human_eval/human_evaluation_template.csv" "$PACKAGE_DIR/human_evaluation/human_evaluation_template.csv"

# ------------------------------------------------------------
# 10. Tests if available
# ------------------------------------------------------------
echo ""
echo "10. Copying tests..."

copy_dir_if_exists "tests" "$PACKAGE_DIR/tests"

# ------------------------------------------------------------
# 11. Create implementation notes file
# ------------------------------------------------------------
echo ""
echo "11. Creating implementation notes..."

cat > "$PACKAGE_DIR/notes/chapter3_notes.txt" <<'EOF'
Chapter 3 Evidence Package
==========================

This package contains selected implementation and result files for:

Chapter 3: Implementation, Experiments, and Results

Recommended chapter structure:
1. Software environment and project structure
2. Data ingestion and preprocessing implementation
3. Chunking and vector-store construction
4. Baseline RAG implementation
5. Agentic RAG v1 implementation
6. Agentic RAG v2 and adaptive routing
7. Experimental setup
8. Evaluation question sets
9. Evaluation metrics
10. Quantitative results
11. Qualitative analysis
12. Baseline RAG vs Agentic RAG comparison
13. Latency and traceability analysis
14. Implementation challenges and solutions
15. Limitations
16. Chapter summary

Important interpretation:
- Baseline RAG performed strongly on direct dataset-aligned questions.
- Agentic RAG mainly improved transparency, source validation, rejected-source tracking,
  sufficiency checking, and complex-query handling.
- Do not overclaim that Agentic RAG is always better.
- The final recommended architecture is adaptive:
  simple query -> Baseline RAG
  complex/risky query -> Agentic RAG
EOF

# ------------------------------------------------------------
# 12. Create ZIP file
# ------------------------------------------------------------
echo ""
echo "12. Creating ZIP file..."

if command -v zip >/dev/null 2>&1; then
    zip -r "$ZIP_NAME" "$PACKAGE_DIR" >/dev/null
    echo "ZIP created with zip command: $ZIP_NAME"
else
    echo "zip command not found. Trying PowerShell Compress-Archive..."

    powershell.exe -NoProfile -Command "Compress-Archive -Path '$PACKAGE_DIR' -DestinationPath '$ZIP_NAME' -Force"

    if [ -f "$ZIP_NAME" ]; then
        echo "ZIP created with PowerShell: $ZIP_NAME"
    else
        echo "Could not create ZIP automatically."
        echo "Please right-click the folder and choose: Send to -> Compressed zipped folder"
    fi
fi

# ------------------------------------------------------------
# 13. Final summary
# ------------------------------------------------------------
echo ""
echo "============================================================"
echo "Chapter 3 evidence package created successfully."
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