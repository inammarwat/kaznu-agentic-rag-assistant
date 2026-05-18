import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_bar_chart(labels, values, title, ylabel, output_path, ylim=None):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(ylabel)

    if ylim:
        plt.ylim(*ylim)
    else:
        max_value = max(values) if values else 1
        plt.ylim(0, max_value * 1.25 if max_value > 0 else 1)

    for i, value in enumerate(values):
        plt.text(i, value, str(value), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def main():
    baseline_eval = read_json(
        Path("outputs/evaluation/baseline_complex30_eval_summary.json")
    )
    baseline_batch = read_json(
        Path("outputs/baseline_rag/baseline_complex30_summary.json")
    )

    agentic_eval = read_json(
        Path("outputs/evaluation/agentic_complex30_eval_summary.json")
    )
    agentic_batch = read_json(
        Path("outputs/agentic_rag/agentic_complex30_summary.json")
    )

    output_dir = Path("outputs/evaluation/plots_complex30")
    systems = ["Baseline RAG", "Agentic RAG"]

    score_metrics = [
        "avg_faithfulness_score",
        "avg_answer_relevance_score",
        "avg_context_relevance_score",
        "avg_completeness_score",
        "avg_citation_quality_score",
        "avg_hallucination_score",
    ]

    for metric in score_metrics:
        values = [
            baseline_eval["overall"][metric],
            agentic_eval["overall"][metric],
        ]

        save_bar_chart(
            labels=systems,
            values=values,
            title=metric.replace("_", " ").title(),
            ylabel="Score (1–5)",
            output_path=output_dir / f"{metric}.png",
            ylim=(0, 5.5),
        )

    save_bar_chart(
        labels=systems,
        values=[
            baseline_eval["hallucination_rate"],
            agentic_eval["hallucination_rate"],
        ],
        title="Hallucination Rate Comparison",
        ylabel="Hallucination Rate",
        output_path=output_dir / "hallucination_rate_comparison.png",
        ylim=(0, 1),
    )

    save_bar_chart(
        labels=systems,
        values=[
            baseline_batch["avg_latency_seconds"],
            agentic_batch["avg_latency_seconds"],
        ],
        title="Average Latency Comparison",
        ylabel="Seconds",
        output_path=output_dir / "latency_comparison.png",
    )

    save_bar_chart(
        labels=[
            "Subqueries / Question",
            "Validated Sources / Question",
            "Rejected Sources / Question",
        ],
        values=[
            agentic_batch.get("avg_subqueries_per_question", 0),
            agentic_batch.get("avg_validated_sources_per_question", 0),
            agentic_batch.get("avg_rejected_sources_per_question", 0),
        ],
        title="Agentic RAG Reasoning Trace Metrics",
        ylabel="Average Count",
        output_path=output_dir / "agentic_trace_metrics.png",
    )

    print("PNG plots saved to:", output_dir)


if __name__ == "__main__":
    main()