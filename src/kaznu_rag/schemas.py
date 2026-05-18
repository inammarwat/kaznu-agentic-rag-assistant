from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import json


@dataclass
class DocumentRecord:
    """
    Standard schema for every extracted knowledge-base unit.

    This object is used for:
    - PDF page text
    - PDF table text
    - Web page text
    - Later chunked records before vectorization
    """

    doc_id: str
    source_type: str
    source_name: str
    content_type: str
    text: str

    url: Optional[str] = None
    page_number: Optional[int] = None
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)