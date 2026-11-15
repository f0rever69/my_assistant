import json

from my_assistant.config.workspace import Workspace
from my_assistant.rag.audit import AuditLog
from my_assistant.rag.index import KnowledgeIndex
from my_assistant.rag.model import MockEvidenceModel
from my_assistant.rag.service import KnowledgeService


def test_audit_tracks_decisions_without_document_or_question_text(tmp_path):
    document = tmp_path / "工资明细.txt"
    private = "私有津贴为300元"
    document.write_text(private, encoding="utf-8")
    index = KnowledgeIndex(tmp_path / "kb.sqlite", Workspace.from_path(tmp_path))
    index.refresh(".")
    [hit] = index.search("津贴")[1]
    log = AuditLog(tmp_path / "audit.sqlite", max_runs=1)
    model = MockEvidenceModel(
        {
            "extract": {
                "topics": [
                    {
                        "question": "津贴",
                        "claims": [
                            {
                                "text": private,
                                "quotes": [
                                    {"chunk_id": hit.chunk_id, "quote": private}
                                ],
                            }
                        ],
                    }
                ]
            },
            "verify": {
                "judgments": [
                    {"claim_id": "1", "verdict": "supported", "evidence_ids": ["1.1"]}
                ]
            },
        }
    )
    service = KnowledgeService(index, model, audit=log)
    first = service.ask("津贴内部问题")
    answer = service.ask("津贴内部问题")
    [report] = log.recent()
    assert first.trace_id != answer.trace_id == report["trace_id"]
    assert report["verdicts"] == {"supported": 1}
    assert report["publication_checks"][0]["status"] == "verified"
    serialized = json.dumps(report, ensure_ascii=False)
    assert private not in serialized
    assert document.name not in serialized
    assert "津贴内部问题" not in serialized
