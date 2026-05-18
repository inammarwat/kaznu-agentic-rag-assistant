import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List

import yaml
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_settings(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def clean_metadata_value(value: Any) -> Any:
    """
    Chroma metadata must be scalar:
    str, int, float, bool, or None.

    Convert complex values to JSON strings.
    """
    if value is None:
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    return json.dumps(value, ensure_ascii=False)


def build_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    base_metadata = {
        "chunk_id": chunk.get("chunk_id", ""),
        "parent_doc_id": chunk.get("parent_doc_id", ""),
        "chunk_index": chunk.get("chunk_index", 0),
        "source_type": chunk.get("source_type", ""),
        "content_type": chunk.get("content_type", ""),
        "source_name": chunk.get("source_name", ""),
        "url": chunk.get("url", ""),
        "page_number": chunk.get("page_number", ""),
        "title": chunk.get("title", ""),
    }

    nested_metadata = chunk.get("metadata") or {}

    merged = {
        **base_metadata,
        **nested_metadata,
    }

    return {
        key: clean_metadata_value(value)
        for key, value in merged.items()
    }


def chunks_to_documents(chunks: List[Dict[str, Any]]) -> tuple[List[Document], List[str]]:
    documents: List[Document] = []
    ids: List[str] = []

    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        chunk_id = chunk.get("chunk_id")

        if not text:
            continue

        if not chunk_id:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata=build_metadata(chunk),
            )
        )
        ids.append(chunk_id)

    return documents, ids


def batch_iter(items: List[Any], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def build_vectorstore(config_path: Path) -> None:
    setup_logging()

    settings = load_settings(config_path)

    paths = settings["paths"]
    embedding_cfg = settings["embedding"]
    vectorstore_cfg = settings["vectorstore"]

    chunks_path = Path(paths["chunks_path"])
    vectorstore_dir = Path(paths["vectorstore_dir"])

    collection_name = vectorstore_cfg["collection_name"]
    batch_size = int(vectorstore_cfg.get("batch_size", 64))
    reset_existing = bool(vectorstore_cfg.get("reset_existing", True))

    if reset_existing and vectorstore_dir.exists():
        logger.info("Removing existing vectorstore directory: %s", vectorstore_dir)
        shutil.rmtree(vectorstore_dir)

    vectorstore_dir.mkdir(parents=True, exist_ok=True)

    chunks = read_jsonl(chunks_path)
    documents, ids = chunks_to_documents(chunks)

    logger.info("Loaded chunks: %s", len(chunks))
    logger.info("Valid documents for vectorstore: %s", len(documents))

    if not documents:
        raise ValueError("No valid documents found for vectorstore creation.")

    logger.info("Loading embedding model: %s", embedding_cfg["model_name"])

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_cfg["model_name"],
        model_kwargs={
            "device": embedding_cfg.get("device", "cpu"),
        },
        encode_kwargs={
            "normalize_embeddings": bool(
                embedding_cfg.get("normalize_embeddings", True)
            )
        },
    )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(vectorstore_dir),
    )

    logger.info("Adding documents to Chroma in batches of %s", batch_size)

    for start, doc_batch in batch_iter(documents, batch_size):
        id_batch = ids[start : start + len(doc_batch)]

        vectorstore.add_documents(
            documents=doc_batch,
            ids=id_batch,
        )

        logger.info(
            "Inserted batch: %s - %s",
            start,
            start + len(doc_batch),
        )

    logger.info("Vectorstore created successfully.")
    logger.info("Persist directory: %s", vectorstore_dir)
    logger.info("Collection name: %s", collection_name)
    logger.info("Total vectors inserted: %s", len(documents))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Chroma vector store from KazNU chunks."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to settings YAML file.",
    )
    args = parser.parse_args()

    build_vectorstore(Path(args.config))


if __name__ == "__main__":
    main()