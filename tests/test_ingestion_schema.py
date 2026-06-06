from kaznu_rag.schemas import DocumentRecord
import json


def test_document_record_serializes_to_json():
    record = DocumentRecord(
        doc_id="test-1",
        source_type="pdf",
        source_name="test.pdf",
        content_type="text",
        text="Sample university policy text.",
        page_number=1,
        metadata={"document_group": "academic_policy"},
    )

    data = json.loads(record.to_json())

    assert data["doc_id"] == "test-1"
    assert data["source_type"] == "pdf"
    assert data["page_number"] == 1
    assert data["metadata"]["document_group"] == "academic_policy"