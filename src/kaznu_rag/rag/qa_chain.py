from pathlib import Path
from typing import Any, Dict

from kaznu_rag.rag.baseline_rag import answer_question


def run_qa(
    question: str,
    config_path: Path = Path("config/settings.yaml"),
    k: int = 5,
) -> Dict[str, Any]:
    return answer_question(
        question=question,
        config_path=config_path,
        k=k,
    )