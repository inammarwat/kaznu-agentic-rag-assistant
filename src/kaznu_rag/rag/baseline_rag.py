import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from kaznu_rag.rag.llm_client import initialize_llm
from kaznu_rag.rag.prompt_templates import (
    BASELINE_RAG_SYSTEM_PROMPT,
    BASELINE_RAG_USER_PROMPT,
)
from kaznu_rag.rag.retriever import retrieve_documents, format_context

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def answer_question(
    question: str,
    config_path: Path = Path("config/settings.yaml"),
    k: int = 5,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    start_time = time.time()

    retrieved_docs = retrieve_documents(
        query=question,
        config_path=config_path,
        k=k,
    )

    context = format_context(retrieved_docs)

    llm = initialize_llm(temperature=temperature)

    user_prompt = BASELINE_RAG_USER_PROMPT.format(
        question=question,
        context=context,
    )

    response = llm.invoke(
        [
            SystemMessage(content=BASELINE_RAG_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]
    )

    latency_seconds = round(time.time() - start_time, 3)

    return {
        "question": question,
        "answer": response.content,
        "retrieved_context": retrieved_docs,
        "k": k,
        "latency_seconds": latency_seconds,
        "mode": "baseline_rag",
    }


def save_output(result: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = (
        result["question"]
        .lower()
        .replace(" ", "_")
        .replace("?", "")
        .replace("/", "_")
    )[:80]

    output_path = output_dir / f"baseline_rag_{safe_name}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return output_path


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Run baseline RAG question answering.")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--config", type=str, default="config/settings.yaml")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="outputs/baseline_rag")

    args = parser.parse_args()

    result = answer_question(
        question=args.question,
        config_path=Path(args.config),
        k=args.k,
    )

    output_path = save_output(result, Path(args.output_dir))

    print("\nANSWER:\n")
    print(result["answer"])

    print("\nRETRIEVED SOURCES:\n")
    for item in result["retrieved_context"]:
        metadata = item["metadata"]
        print(
            f"{item['rank']}. score={item['score']:.4f} | "
            f"{metadata.get('content_type')} | "
            f"{metadata.get('source_name')}"
        )

    print(f"\nSaved output: {output_path}")


if __name__ == "__main__":
    main()