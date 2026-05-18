import yaml
from pathlib import Path
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_settings(config_path: Path) -> Dict[str, Any]:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def build_embeddings(settings: Dict[str, Any]) -> HuggingFaceEmbeddings:
    embedding_cfg = settings["embedding"]

    return HuggingFaceEmbeddings(
        model_name=embedding_cfg["model_name"],
        model_kwargs={"device": embedding_cfg.get("device", "cpu")},
        encode_kwargs={
            "normalize_embeddings": bool(
                embedding_cfg.get("normalize_embeddings", True)
            )
        },
    )


def load_vectorstore(config_path: Path) -> Chroma:
    settings = load_settings(config_path)
    embeddings = build_embeddings(settings)

    return Chroma(
        collection_name=settings["vectorstore"]["collection_name"],
        embedding_function=embeddings,
        persist_directory=settings["paths"]["vectorstore_dir"],
    )


def retrieve_documents(
    query: str,
    config_path: Path = Path("config/settings.yaml"),
    k: int = 5,
) -> List[Dict[str, Any]]:
    db = load_vectorstore(config_path)

    results = db.similarity_search_with_score(query, k=k)

    retrieved = []

    for rank, (doc, score) in enumerate(results, start=1):
        retrieved.append(
            {
                "rank": rank,
                "score": float(score),
                "text": doc.page_content,
                "metadata": dict(doc.metadata),
            }
        )

    return retrieved


def format_source_label(metadata: Dict[str, Any]) -> str:
    source_name = metadata.get("source_name", "Unknown source")
    page_number = metadata.get("page_number", "")
    url = metadata.get("url", "")
    content_type = metadata.get("content_type", "")

    if url:
        return f"{source_name}"

    if page_number not in {"", None}:
        return f"{source_name}, page {page_number}"

    if content_type == "tuition_fee_fact":
        return f"{source_name}, structured tuition record"

    return source_name


def format_context(retrieved_docs: List[Dict[str, Any]]) -> str:
    context_blocks = []

    for item in retrieved_docs:
        metadata = item["metadata"]
        source_label = format_source_label(metadata)

        context_blocks.append(
            f"[Source {item['rank']}: {source_label}]\n"
            f"{item['text']}"
        )

    return "\n\n".join(context_blocks)