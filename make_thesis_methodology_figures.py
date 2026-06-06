from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("outputs/thesis_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DPI = 300

TITLE_FONT = 16
SUBTITLE_FONT = 11
BOX_FONT = 10
SMALL_FONT = 9


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def wrap_text(text: str, width: int = 22) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def draw_box(
    ax,
    x,
    y,
    w,
    h,
    text,
    facecolor="#EAF2F8",
    edgecolor="#1F4E79",
    fontsize=BOX_FONT,
    linewidth=1.8,
    roundness=0.02,
    text_color="black",
):
    """
    Draw a rounded rectangle box.
    Coordinates are in normalized figure space [0,1].
    """
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.01,rounding_size={roundness}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        wrap_text(text, 22),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight="bold",
    )


def draw_arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.8, mutation_scale=16):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        shrinkA=0,
        shrinkB=0,
    )
    ax.add_patch(arrow)


def setup_canvas(title: str, subtitle: str = ""):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.95, title,
        ha="center", va="center",
        fontsize=TITLE_FONT,
        fontweight="bold"
    )

    if subtitle:
        ax.text(
            0.5, 0.90, subtitle,
            ha="center", va="center",
            fontsize=SUBTITLE_FONT,
            style="italic",
            color="dimgray"
        )

    return fig, ax


def save_figure(fig, filename_stem: str):
    png_path = OUTPUT_DIR / f"{filename_stem}.png"
    svg_path = OUTPUT_DIR / f"{filename_stem}.svg"

    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {svg_path}")


# ============================================================
# FIGURE 2.1
# Data ingestion workflow
# ============================================================

def figure_2_1_data_ingestion_workflow():
    fig, ax = setup_canvas(
        "Figure 2.1 — Data Ingestion Workflow",
        "Raw PDFs and web URLs are converted into cleaned records and chunked for retrieval."
    )

    y = 0.45
    w = 0.13
    h = 0.18

    xs = [0.03, 0.20, 0.37, 0.54, 0.71, 0.86]

    labels = [
        "Raw PDFs",
        "Web URLs",
        "PDF Extraction",
        "Web Extraction",
        "Raw Records → Cleaned Records",
        "Chunking"
    ]

    colors = [
        "#D6EAF8",
        "#D6EAF8",
        "#FCF3CF",
        "#FCF3CF",
        "#E8F8F5",
        "#FDEDEC",
    ]

    for i, (x, label, color) in enumerate(zip(xs, labels, colors)):
        draw_box(ax, x, y, w, h, label, facecolor=color)

    # arrows
    draw_arrow(ax, xs[0] + w, y + h / 2, xs[2], y + h / 2)
    draw_arrow(ax, xs[1] + w, y + h / 2, xs[3], y + h / 2)
    draw_arrow(ax, xs[2] + w, y + h / 2, xs[4], y + h / 2 + 0.06)
    draw_arrow(ax, xs[3] + w, y + h / 2, xs[4], y + h / 2 - 0.06)
    draw_arrow(ax, xs[4] + w, y + h / 2, xs[5], y + h / 2)

    save_figure(fig, "figure_2_1_data_ingestion_workflow")


# ============================================================
# FIGURE 2.2
# Tuition-fee table extraction and normalisation pipeline
# ============================================================

def figure_2_2_tuition_normalisation_pipeline():
    fig, ax = setup_canvas(
        "Figure 2.2 — Tuition-Fee Table Extraction and Normalisation Pipeline",
        "Tuition data is extracted from the PDF, cleaned, normalised, and converted into one-fact-per-chunk records."
    )

    y = 0.45
    w = 0.14
    h = 0.18

    xs = [0.02, 0.19, 0.36, 0.53, 0.70, 0.86]

    labels = [
        "Tuition PDF",
        "Table Extraction",
        "Row Cleaning",
        "Attribute Normalisation",
        "One Fact per Chunk",
        "Metadata Enrichment"
    ]

    colors = [
        "#D6EAF8",
        "#FCF3CF",
        "#E8F8F5",
        "#F5EEF8",
        "#FDEDEC",
        "#EAF2F8"
    ]

    for x, label, color in zip(xs, labels, colors):
        draw_box(ax, x, y, w, h, label, facecolor=color)

    for i in range(len(xs) - 1):
        draw_arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    save_figure(fig, "figure_2_2_tuition_normalisation_pipeline")


# ============================================================
# FIGURE 2.3
# Embedding and vector-store construction
# ============================================================

def figure_2_3_embedding_vectorstore_construction():
    fig, ax = setup_canvas(
        "Figure 2.3 — Embedding and Vector-Store Construction",
        "Clean text chunks and metadata are transformed into embeddings and stored in ChromaDB for retrieval."
    )

    y = 0.45
    w = 0.14
    h = 0.18

    xs = [0.03, 0.22, 0.41, 0.60, 0.79]

    labels = [
        "Clean Chunks + Metadata",
        "Embedding Model\n(BAAI/bge-small-en-v1.5)",
        "Vectors",
        "ChromaDB",
        "Retriever"
    ]

    colors = [
        "#E8F8F5",
        "#FCF3CF",
        "#FDEDEC",
        "#D6EAF8",
        "#F5EEF8"
    ]

    for x, label, color in zip(xs, labels, colors):
        draw_box(ax, x, y, w, h, label, facecolor=color)

    for i in range(len(xs) - 1):
        draw_arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    save_figure(fig, "figure_2_3_embedding_vectorstore_construction")


# ============================================================
# FIGURE 2.4
# Baseline RAG workflow
# ============================================================

def figure_2_4_baseline_rag_workflow():
    fig, ax = setup_canvas(
        "Figure 2.4 — Baseline RAG Workflow",
        "A direct retrieval-augmented generation pipeline for standard university-information questions."
    )

    y = 0.45
    w = 0.12
    h = 0.18

    xs = [0.01, 0.15, 0.29, 0.43, 0.57, 0.71, 0.85]

    labels = [
        "User Question",
        "Query Embedding",
        "Chroma Similarity Search",
        "Top-k Chunks",
        "Prompt",
        "LLM",
        "Answer with Citations"
    ]

    colors = [
        "#D6EAF8",
        "#FCF3CF",
        "#E8F8F5",
        "#FDEDEC",
        "#F5EEF8",
        "#D6EAF8",
        "#EAF2F8"
    ]

    for x, label, color in zip(xs, labels, colors):
        draw_box(ax, x, y, w, h, label, facecolor=color)

    for i in range(len(xs) - 1):
        draw_arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    save_figure(fig, "figure_2_4_baseline_rag_workflow")


# ============================================================
# FIGURE 2.5
# Agentic RAG v1 workflow
# ============================================================

def figure_2_5_agentic_rag_v1_workflow():
    fig, ax = setup_canvas(
        "Figure 2.5 — Agentic RAG v1 Workflow",
        "Complex questions are decomposed, retrieved through multiple subqueries, fused, validated, and answered."
    )

    y = 0.45
    w = 0.11
    h = 0.18

    xs = [0.01, 0.14, 0.27, 0.40, 0.53, 0.66, 0.79, 0.90]

    labels = [
        "User Question",
        "Decomposition",
        "Subqueries",
        "Multi-query Retrieval",
        "Reciprocal-rank Fusion",
        "Source Validation",
        "Validated Context",
        "LLM Answer"
    ]

    colors = [
        "#D6EAF8",
        "#FCF3CF",
        "#F5EEF8",
        "#E8F8F5",
        "#FDEDEC",
        "#FCF3CF",
        "#EAF2F8",
        "#D6EAF8"
    ]

    for x, label, color in zip(xs, labels, colors):
        draw_box(ax, x, y, w, h, label, facecolor=color, fontsize=9)

    for i in range(len(xs) - 1):
        draw_arrow(ax, xs[i] + w, y + h / 2, xs[i + 1], y + h / 2)

    save_figure(fig, "figure_2_5_agentic_rag_v1_workflow")


# ============================================================
# FIGURE 2.6
# Agentic RAG v2 and adaptive routing architecture
# ============================================================

def figure_2_6_agentic_rag_v2_adaptive_routing():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.95,
        "Figure 2.6 — Agentic RAG v2 and Adaptive Routing Architecture",
        ha="center", va="center", fontsize=TITLE_FONT, fontweight="bold"
    )
    ax.text(
        0.5, 0.91,
        "Simple questions are routed to Baseline RAG, while complex questions are routed to Agentic RAG, followed by sufficiency scoring and reflection.",
        ha="center", va="center", fontsize=SUBTITLE_FONT, style="italic", color="dimgray"
    )

    # top
    draw_box(ax, 0.40, 0.78, 0.20, 0.10, "User Question", facecolor="#D6EAF8")
    draw_box(ax, 0.37, 0.60, 0.26, 0.10, "Query Classifier / Router", facecolor="#FCF3CF")

    # branches
    draw_box(ax, 0.12, 0.40, 0.25, 0.12, "Baseline RAG\n(for Simple Questions)", facecolor="#E8F8F5")
    draw_box(ax, 0.63, 0.40, 0.25, 0.12, "Agentic RAG\n(for Complex Questions)", facecolor="#FDEDEC")

    # lower steps
    draw_box(ax, 0.37, 0.22, 0.26, 0.10, "Sufficiency Scoring", facecolor="#F5EEF8")
    draw_box(ax, 0.37, 0.08, 0.26, 0.10, "Reflection", facecolor="#EAF2F8")
    draw_box(ax, 0.37, 0.00, 0.26, 0.06, "Final Answer", facecolor="#D6EAF8")

    # arrows
    draw_arrow(ax, 0.50, 0.78, 0.50, 0.70)
    draw_arrow(ax, 0.50, 0.60, 0.245, 0.52)
    draw_arrow(ax, 0.50, 0.60, 0.755, 0.52)

    draw_arrow(ax, 0.245, 0.40, 0.50, 0.32)
    draw_arrow(ax, 0.755, 0.40, 0.50, 0.32)

    draw_arrow(ax, 0.50, 0.22, 0.50, 0.18)
    draw_arrow(ax, 0.50, 0.08, 0.50, 0.06)

    # notes
    ax.text(0.24, 0.35, "Simple / direct / single-intent", ha="center", fontsize=SMALL_FONT, color="dimgray")
    ax.text(0.76, 0.35, "Complex / multi-source / high-risk", ha="center", fontsize=SMALL_FONT, color="dimgray")

    save_figure(fig, "figure_2_6_agentic_rag_v2_adaptive_routing")


# ============================================================
# MAIN
# ============================================================

def main():
    figure_2_1_data_ingestion_workflow()
    figure_2_2_tuition_normalisation_pipeline()
    figure_2_3_embedding_vectorstore_construction()
    figure_2_4_baseline_rag_workflow()
    figure_2_5_agentic_rag_v1_workflow()
    figure_2_6_agentic_rag_v2_adaptive_routing()

    print("\nAll thesis methodology figures generated successfully.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()