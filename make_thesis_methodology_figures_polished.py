"""
Generate polished methodology figures for the MSc thesis.

Project:
KazNU Agentic RAG Assistant

Output:
outputs/thesis_figures_polished/

Generated files:
- data_ingestion_workflow.png / .svg / .pdf
- tuition_fee_normalisation_pipeline.png / .svg / .pdf
- embedding_vectorstore_construction.png / .svg / .pdf
- baseline_rag_workflow.png / .svg / .pdf
- agentic_rag_v1_workflow.png / .svg / .pdf
- adaptive_agentic_rag_v2_architecture.png / .svg / .pdf

Note:
The figure numbers are intentionally NOT placed inside the diagrams.
Use figure numbers only in MS Word captions.
"""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("outputs/thesis_figures_polished")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 300

plt.rcParams["figure.dpi"] = DPI
plt.rcParams["savefig.dpi"] = DPI
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#2F3E46"


# ============================================================
# ACADEMIC COLOR PALETTE
# ============================================================

COLORS = {
    "source": "#DCEEF8",
    "ingestion": "#FFF3D6",
    "cleaning": "#E1F3E8",
    "structure": "#EDE3F4",
    "embedding": "#FFF8D6",
    "vector": "#E2EDFF",
    "baseline": "#DDEEFF",
    "agentic": "#FDE2E2",
    "validation": "#E7F5DF",
    "reflection": "#E9E5F6",
    "routing": "#E5E7E9",
    "output": "#E8EEF8",
    "white": "#FFFFFF",
    "dark": "#2F3E46",
    "muted": "#607D8B",
    "line": "#526D82",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def wrap(text: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width))


def setup_canvas(title: str, subtitle: str = "", figsize=(16, 5.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Outer light frame
    frame = FancyBboxPatch(
        (0.01, 0.02),
        0.98,
        0.94,
        boxstyle="round,pad=0.004,rounding_size=0.018",
        linewidth=1.1,
        edgecolor="#D0D7DE",
        facecolor="#FFFFFF",
        zorder=0,
    )
    ax.add_patch(frame)

    ax.text(
        0.5,
        0.925,
        title,
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=COLORS["dark"],
    )

    if subtitle:
        ax.text(
            0.5,
            0.875,
            subtitle,
            ha="center",
            va="center",
            fontsize=10.5,
            color=COLORS["muted"],
            style="italic",
        )

    return fig, ax


def draw_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    detail=None,
    facecolor="#EAF2F8",
    edgecolor="#405D72",
    title_size=10.5,
    detail_size=8.5,
    shadow=True,
):
    """Draw a rounded academic-style box with optional detail text."""

    if shadow:
        shadow_patch = FancyBboxPatch(
            (x + 0.006, y - 0.006),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0,
            facecolor="#D9DEE3",
            alpha=0.35,
            zorder=1,
        )
        ax.add_patch(shadow_patch)

    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2,
    )
    ax.add_patch(box)

    if detail:
        ax.text(
            x + w / 2,
            y + h * 0.63,
            wrap(title, 22),
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=COLORS["dark"],
            zorder=3,
        )
        ax.text(
            x + w / 2,
            y + h * 0.30,
            wrap(detail, 28),
            ha="center",
            va="center",
            fontsize=detail_size,
            color="#37474F",
            zorder=3,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            wrap(title, 22),
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=COLORS["dark"],
            zorder=3,
        )


def draw_arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    color=None,
    lw=1.8,
    curve=0.0,
    label=None,
    label_offset=(0, 0),
):
    """Draw a clean arrow, optionally curved."""

    if color is None:
        color = COLORS["line"]

    connectionstyle = f"arc3,rad={curve}"

    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        zorder=4,
    )
    ax.add_patch(arrow)

    if label:
        ax.text(
            (x1 + x2) / 2 + label_offset[0],
            (y1 + y2) / 2 + label_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color=color,
            fontweight="bold",
            zorder=5,
        )


def draw_step_badge(ax, x, y, number, color="#405D72"):
    circle = Circle((x, y), 0.025, facecolor=color, edgecolor="white", linewidth=1.2, zorder=6)
    ax.add_patch(circle)
    ax.text(
        x,
        y,
        str(number),
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold",
        zorder=7,
    )


def draw_linear_flow(title, subtitle, steps, filename, figsize=(16, 5.2)):
    """Draw a polished left-to-right linear workflow."""

    fig, ax = setup_canvas(title, subtitle, figsize=figsize)

    n = len(steps)
    margin_left = 0.045
    margin_right = 0.045
    gap = 0.025
    usable = 1 - margin_left - margin_right
    w = (usable - gap * (n - 1)) / n
    h = 0.22
    y = 0.43

    xs = [margin_left + i * (w + gap) for i in range(n)]

    for idx, (x, step) in enumerate(zip(xs, steps), start=1):
        draw_box(
            ax,
            x,
            y,
            w,
            h,
            step["title"],
            step.get("detail"),
            facecolor=step.get("color", COLORS["white"]),
            edgecolor=step.get("edge", "#405D72"),
            title_size=step.get("title_size", 9.8),
            detail_size=step.get("detail_size", 7.8),
        )
        draw_step_badge(ax, x + 0.025, y + h + 0.035, idx)

    for i in range(n - 1):
        draw_arrow(
            ax,
            xs[i] + w,
            y + h / 2,
            xs[i + 1],
            y + h / 2,
        )

    save_all(fig, filename)


def save_all(fig, filename_stem):
    png_path = OUTPUT_DIR / f"{filename_stem}.png"
    svg_path = OUTPUT_DIR / f"{filename_stem}.svg"
    pdf_path = OUTPUT_DIR / f"{filename_stem}.pdf"

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")
    print(f"Saved: {pdf_path}")


# ============================================================
# FIGURE 1: DATA INGESTION WORKFLOW
# ============================================================

def make_data_ingestion_workflow():
    fig, ax = setup_canvas(
        "Data Ingestion Workflow",
        "PDF and web sources are extracted, cleaned, and converted into retrieval-ready chunks.",
        figsize=(16, 6),
    )

    # Source boxes
    draw_box(
        ax,
        0.06,
        0.62,
        0.18,
        0.18,
        "Raw PDF Sources",
        "Academic policy, booklet, tuition fee, AI regulation",
        facecolor=COLORS["source"],
    )
    draw_box(
        ax,
        0.06,
        0.30,
        0.18,
        0.18,
        "Web URL Sources",
        "KazNU and Farabi university webpages",
        facecolor=COLORS["source"],
    )

    # Extraction boxes
    draw_box(
        ax,
        0.32,
        0.62,
        0.18,
        0.18,
        "PDF Extraction",
        "Page-wise text and table extraction",
        facecolor=COLORS["ingestion"],
    )
    draw_box(
        ax,
        0.32,
        0.30,
        0.18,
        0.18,
        "Web Extraction",
        "HTML fetching and main-content extraction",
        facecolor=COLORS["ingestion"],
    )

    # Merge and clean
    draw_box(
        ax,
        0.58,
        0.47,
        0.17,
        0.20,
        "Raw Records",
        "PDF text, PDF tables, and web records",
        facecolor=COLORS["cleaning"],
    )
    draw_box(
        ax,
        0.80,
        0.47,
        0.15,
        0.20,
        "Cleaned Records",
        "Noise removal, normalisation, source metadata",
        facecolor=COLORS["structure"],
    )

    draw_box(
        ax,
        0.80,
        0.18,
        0.15,
        0.16,
        "Chunking",
        "Retrieval-ready text units",
        facecolor=COLORS["output"],
    )

    # Badges
    draw_step_badge(ax, 0.06, 0.83, 1)
    draw_step_badge(ax, 0.32, 0.83, 2)
    draw_step_badge(ax, 0.58, 0.70, 3)
    draw_step_badge(ax, 0.80, 0.70, 4)
    draw_step_badge(ax, 0.80, 0.37, 5)

    # Arrows
    draw_arrow(ax, 0.24, 0.71, 0.32, 0.71)
    draw_arrow(ax, 0.24, 0.39, 0.32, 0.39)

    draw_arrow(ax, 0.50, 0.71, 0.58, 0.60, curve=-0.10)
    draw_arrow(ax, 0.50, 0.39, 0.58, 0.54, curve=0.10)

    draw_arrow(ax, 0.75, 0.57, 0.80, 0.57)
    draw_arrow(ax, 0.875, 0.47, 0.875, 0.34)

    save_all(fig, "data_ingestion_workflow")


# ============================================================
# FIGURE 2: TUITION NORMALISATION PIPELINE
# ============================================================

def make_tuition_normalisation_pipeline():
    steps = [
        {
            "title": "Tuition Fee PDF",
            "detail": "Raw tabular tuition source",
            "color": COLORS["source"],
        },
        {
            "title": "Table Extraction",
            "detail": "Extract tables from PDF pages",
            "color": COLORS["ingestion"],
        },
        {
            "title": "Row Cleaning",
            "detail": "Remove broken rows and artefacts",
            "color": COLORS["cleaning"],
        },
        {
            "title": "Attribute Normalisation",
            "detail": "Faculty, degree, language, region, year, fee",
            "color": COLORS["structure"],
        },
        {
            "title": "One Fact per Chunk",
            "detail": "Convert each normalised fee row into one retrievable fact",
            "color": COLORS["validation"],
        },
        {
            "title": "Metadata Enrichment",
            "detail": "Attach source, page, academic year, and constraints",
            "color": COLORS["output"],
        },
    ]

    draw_linear_flow(
        "Tuition-Fee Table Extraction and Normalisation Pipeline",
        "Structured tuition information is converted into constraint-aware retrieval units.",
        steps,
        "tuition_fee_normalisation_pipeline",
        figsize=(17, 5.4),
    )


# ============================================================
# FIGURE 3: EMBEDDING AND VECTOR STORE
# ============================================================

def make_embedding_vectorstore_construction():
    steps = [
        {
            "title": "Clean Chunks + Metadata",
            "detail": "PDF, web, table, and structured facts",
            "color": COLORS["cleaning"],
        },
        {
            "title": "Embedding Model",
            "detail": "BAAI/bge-small-en-v1.5",
            "color": COLORS["embedding"],
        },
        {
            "title": "Dense Vectors",
            "detail": "Semantic vector representations",
            "color": COLORS["structure"],
        },
        {
            "title": "ChromaDB Vector Store",
            "detail": "Persistent vector database with metadata",
            "color": COLORS["vector"],
        },
        {
            "title": "Retriever",
            "detail": "Similarity search for RAG pipelines",
            "color": COLORS["output"],
        },
    ]

    draw_linear_flow(
        "Embedding and Vector-Store Construction",
        "Metadata-rich chunks are embedded and stored for similarity-based retrieval.",
        steps,
        "embedding_vectorstore_construction",
        figsize=(16, 5.2),
    )


# ============================================================
# FIGURE 4: BASELINE RAG WORKFLOW
# ============================================================

def make_baseline_rag_workflow():
    steps = [
        {
            "title": "User Question",
            "detail": "Direct information need",
            "color": COLORS["source"],
        },
        {
            "title": "Query Embedding",
            "detail": "Encode question with same embedding model",
            "color": COLORS["embedding"],
        },
        {
            "title": "Chroma Similarity Search",
            "detail": "Retrieve semantically similar chunks",
            "color": COLORS["vector"],
        },
        {
            "title": "Top-k Chunks",
            "detail": "k = 5 retrieved evidence units",
            "color": COLORS["cleaning"],
        },
        {
            "title": "Prompt Construction",
            "detail": "Question + retrieved context",
            "color": COLORS["structure"],
        },
        {
            "title": "LLM Generation",
            "detail": "Answer only from context",
            "color": COLORS["baseline"],
        },
        {
            "title": "Answer with Citations",
            "detail": "Grounded response using source markers",
            "color": COLORS["output"],
        },
    ]

    draw_linear_flow(
        "Baseline RAG Workflow",
        "A single-query retrieve-then-generate pipeline for direct university information questions.",
        steps,
        "baseline_rag_workflow",
        figsize=(18, 5.4),
    )


# ============================================================
# FIGURE 5: AGENTIC RAG V1 WORKFLOW
# ============================================================

def make_agentic_rag_v1_workflow():
    steps = [
        {
            "title": "User Question",
            "detail": "Complex or multi-part information need",
            "color": COLORS["source"],
            "title_size": 9.4,
        },
        {
            "title": "Query Decomposition",
            "detail": "Break question into focused subqueries",
            "color": COLORS["ingestion"],
            "title_size": 9.4,
        },
        {
            "title": "Subqueries",
            "detail": "Multiple retrieval-oriented queries",
            "color": COLORS["structure"],
            "title_size": 9.4,
        },
        {
            "title": "Multi-query Retrieval",
            "detail": "Retrieve evidence for each subquery",
            "color": COLORS["vector"],
            "title_size": 9.4,
        },
        {
            "title": "Reciprocal-Rank Fusion",
            "detail": "Merge and rank retrieved candidates",
            "color": COLORS["embedding"],
            "title_size": 9.4,
        },
        {
            "title": "Source Validation",
            "detail": "Filter by relevance and constraints",
            "color": COLORS["validation"],
            "title_size": 9.4,
        },
        {
            "title": "Validated Context",
            "detail": "Accepted evidence for generation",
            "color": COLORS["cleaning"],
            "title_size": 9.4,
        },
        {
            "title": "LLM Answer",
            "detail": "Grounded response with citations",
            "color": COLORS["agentic"],
            "title_size": 9.4,
        },
    ]

    draw_linear_flow(
        "Agentic RAG v1 Workflow",
        "Complex queries are decomposed, retrieved through multiple paths, fused, validated, and answered.",
        steps,
        "agentic_rag_v1_workflow",
        figsize=(19, 5.5),
    )


# ============================================================
# FIGURE 6: ADAPTIVE AGENTIC RAG V2 ARCHITECTURE
# ============================================================

def make_adaptive_agentic_rag_v2_architecture():
    """
    Clean, thesis-ready Figure 2.6.

    This version intentionally uses short labels only.
    Detailed explanation should be written in the thesis caption/text,
    not inside the boxes.
    """

    fig, ax = setup_canvas(
        "Agentic RAG v2 and Adaptive Routing Architecture",
        "Simple queries are routed to Baseline RAG; complex queries are routed to Agentic RAG with evidence-control steps.",
        figsize=(16, 8.5),
    )

    # ========================================================
    # Top layer
    # ========================================================

    draw_box(
        ax,
        0.39,
        0.78,
        0.22,
        0.10,
        "User Question",
        facecolor=COLORS["source"],
        edgecolor="#405D72",
        title_size=11.5,
    )

    draw_box(
        ax,
        0.36,
        0.62,
        0.28,
        0.11,
        "Query Classifier / Router",
        facecolor=COLORS["routing"],
        edgecolor="#405D72",
        title_size=11.0,
    )

    draw_arrow(ax, 0.50, 0.78, 0.50, 0.73)

    # ========================================================
    # Branch labels
    # ========================================================

    ax.text(
        0.24,
        0.56,
        "Simple query path",
        ha="center",
        va="center",
        fontsize=11,
        color="#2E75B6",
        fontweight="bold",
    )

    ax.text(
        0.76,
        0.56,
        "Complex query path",
        ha="center",
        va="center",
        fontsize=11,
        color="#C0392B",
        fontweight="bold",
    )

    # ========================================================
    # Baseline RAG branch
    # ========================================================

    draw_box(
        ax,
        0.08,
        0.41,
        0.32,
        0.11,
        "Baseline RAG",
        facecolor=COLORS["baseline"],
        edgecolor="#2E75B6",
        title_size=11.0,
    )

    draw_box(
        ax,
        0.08,
        0.27,
        0.32,
        0.10,
        "Direct Answer",
        facecolor=COLORS["output"],
        edgecolor="#2E75B6",
        title_size=10.8,
    )

    # ========================================================
    # Agentic RAG branch
    # ========================================================

    draw_box(
        ax,
        0.60,
        0.44,
        0.32,
        0.10,
        "Agentic RAG v1",
        facecolor=COLORS["agentic"],
        edgecolor="#C0392B",
        title_size=11.0,
    )

    draw_box(
        ax,
        0.60,
        0.31,
        0.32,
        0.10,
        "Source Sufficiency Scoring",
        facecolor=COLORS["validation"],
        edgecolor="#6AA84F",
        title_size=10.7,
    )

    draw_box(
        ax,
        0.60,
        0.18,
        0.32,
        0.10,
        "Reflection Agent",
        facecolor=COLORS["reflection"],
        edgecolor="#7E57C2",
        title_size=11.0,
    )

    # ========================================================
    # Final answer
    # ========================================================

    draw_box(
        ax,
        0.34,
        0.06,
        0.32,
        0.09,
        "Final Answer with Citations",
        facecolor=COLORS["output"],
        edgecolor="#405D72",
        title_size=10.8,
    )

    # ========================================================
    # Router arrows
    # ========================================================

    draw_arrow(
        ax,
        0.36,
        0.665,
        0.24,
        0.52,
        color="#2E75B6",
        label="Simple",
        label_offset=(-0.035, 0.015),
    )

    draw_arrow(
        ax,
        0.64,
        0.665,
        0.76,
        0.54,
        color="#C0392B",
        label="Complex",
        label_offset=(0.035, 0.015),
    )

    # ========================================================
    # Baseline path arrows
    # ========================================================

    draw_arrow(
        ax,
        0.24,
        0.41,
        0.24,
        0.37,
        color="#2E75B6",
    )

    draw_arrow(
        ax,
        0.24,
        0.27,
        0.40,
        0.15,
        curve=-0.08,
        color="#2E75B6",
    )

    # ========================================================
    # Agentic path arrows
    # ========================================================

    draw_arrow(
        ax,
        0.76,
        0.44,
        0.76,
        0.41,
        color="#C0392B",
    )

    draw_arrow(
        ax,
        0.76,
        0.31,
        0.76,
        0.28,
        color="#6AA84F",
    )

    draw_arrow(
        ax,
        0.76,
        0.18,
        0.60,
        0.15,
        curve=0.05,
        color="#7E57C2",
    )

    # ========================================================
    # Small explanatory labels below branches
    # ========================================================

    ax.text(
        0.24,
        0.225,
        "Fast path for direct factual questions",
        ha="center",
        va="center",
        fontsize=8.8,
        color=COLORS["muted"],
        style="italic",
    )

    ax.text(
        0.76,
        0.135,
        "Evidence-control path for complex or risky questions",
        ha="center",
        va="center",
        fontsize=8.8,
        color=COLORS["muted"],
        style="italic",
    )

    # ========================================================
    # Design principle note
    # ========================================================

    ax.text(
        0.50,
        0.025,
        "Design principle: use Baseline RAG for efficiency and Agentic RAG for decomposition, validation, sufficiency checking, and reflection.",
        ha="center",
        va="center",
        fontsize=8.5,
        color=COLORS["muted"],
        style="italic",
    )

    save_all(fig, "adaptive_agentic_rag_v2_architecture")


# ============================================================
# MAIN
# ============================================================

def main():
    make_data_ingestion_workflow()
    make_tuition_normalisation_pipeline()
    make_embedding_vectorstore_construction()
    make_baseline_rag_workflow()
    make_agentic_rag_v1_workflow()
    make_adaptive_agentic_rag_v2_architecture()

    print("\nAll polished methodology figures generated successfully.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()