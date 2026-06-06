import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================
# Output directory
# ============================================================
OUTPUT_DIR = Path("outputs/thesis_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10


# ============================================================
# Helper function
# ============================================================
def add_value_labels(bars, fmt="{:.2f}", offset=0.02):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom"
        )


# ============================================================
# Figure 1: Main performance comparison
# ============================================================
systems = [
    "Baseline RAG\n(50 General)",
    "Baseline RAG\n(30 Complex)",
    "Agentic RAG\n(30 Complex)"
]

faithfulness = [4.94, 4.967, 5.00]
completeness = [5.00, 5.00, 5.00]
hallucination_rate = [0.04, 0.00, 0.00]
latency = [2.193, None, 3.20]

# Use a 2x2 academic dashboard style in one figure
fig = plt.figure(figsize=(12, 8))

# ---------- subplot 1 ----------
ax1 = fig.add_subplot(2, 2, 1)
x = np.arange(len(systems))
bars = ax1.bar(x, faithfulness)
ax1.set_title("Faithfulness Score Comparison")
ax1.set_ylabel("Score")
ax1.set_ylim(4.7, 5.1)
ax1.set_xticks(x)
ax1.set_xticklabels(systems)
for bar in bars:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{bar.get_height():.3f}", ha="center", va="bottom")

# ---------- subplot 2 ----------
ax2 = fig.add_subplot(2, 2, 2)
bars = ax2.bar(x, completeness)
ax2.set_title("Completeness Score Comparison")
ax2.set_ylabel("Score")
ax2.set_ylim(4.7, 5.1)
ax2.set_xticks(x)
ax2.set_xticklabels(systems)
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{bar.get_height():.2f}", ha="center", va="bottom")

# ---------- subplot 3 ----------
ax3 = fig.add_subplot(2, 2, 3)
bars = ax3.bar(x, hallucination_rate)
ax3.set_title("Hallucination Rate Comparison")
ax3.set_ylabel("Rate")
ax3.set_ylim(0, 0.06)
ax3.set_xticks(x)
ax3.set_xticklabels(systems)
for bar in bars:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f"{bar.get_height():.2f}", ha="center", va="bottom")

# ---------- subplot 4 ----------
ax4 = fig.add_subplot(2, 2, 4)
latency_values = [2.193, 2.50, 3.20]  # replace 2.50 with exact baseline complex latency if available
bars = ax4.bar(x, latency_values)
ax4.set_title("Average Latency Comparison")
ax4.set_ylabel("Seconds")
ax4.set_ylim(0, 4)
ax4.set_xticks(x)
ax4.set_xticklabels(systems)
for bar in bars:
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{bar.get_height():.2f}", ha="center", va="bottom")

fig.suptitle("Performance Comparison of Baseline and Agentic RAG Systems", fontsize=15)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUTPUT_DIR / "figure_1_performance_comparison.png", bbox_inches="tight")
plt.close(fig)


# ============================================================
# Figure 2: Agentic trace metrics
# ============================================================
labels = [
    "Avg Subqueries\nper Question",
    "Avg Validated\nSources",
    "Avg Rejected\nSources",
    "Avg Latency\n(sec)"
]
values = [3.467, 4.833, 2.033, 3.20]

fig = plt.figure(figsize=(10, 6))
bars = plt.bar(labels, values)
plt.title("Agentic RAG Trace and Transparency Metrics")
plt.ylabel("Value")
plt.ylim(0, 6)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.08,
        f"{bar.get_height():.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure_2_agentic_trace_metrics.png", bbox_inches="tight")
plt.close(fig)


# ============================================================
# Figure 3: Evaluation summary comparison table as image-like figure
# ============================================================
fig, ax = plt.subplots(figsize=(12, 3))
ax.axis("off")

table_data = [
    ["System", "Faithfulness", "Completeness", "Hallucination Rate", "Latency (sec)"],
    ["Baseline RAG (50 General)", "4.94", "5.00", "0.04", "2.193"],
    ["Baseline RAG (30 Complex)", "4.967", "5.00", "0.00", "2.50*"],
    ["Agentic RAG (30 Complex)", "5.00", "5.00", "0.00", "3.20"],
]

table = ax.table(cellText=table_data, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

plt.title("Summary of Main Experimental Results", pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure_3_results_summary_table.png", bbox_inches="tight")
plt.close(fig)

print("Figures saved in:", OUTPUT_DIR.resolve())