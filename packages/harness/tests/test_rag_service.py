from my_assistant.config.workspace import Workspace
from my_assistant.rag.index import KnowledgeIndex
from my_assistant.rag.model import MockEvidenceModel
from my_assistant.rag.service import KnowledgeService


def _fixture(tmp_path):
    document = tmp_path / "差旅报销.txt"
    quote = "出差结束后30天内提交报销。"
    document.write_text(quote, encoding="utf-8")
    index = KnowledgeIndex(tmp_path / "kb.sqlite", Workspace.from_path(tmp_path))
    index.refresh(".")
    hit = index.search("报销")[1][0]
    source = {"chunk_id": hit.chunk_id, "quote": quote}
    return index, document, source


def test_verified_batch_keeps_unanswered_topics_and_hides_wrong_facts(tmp_path):
    index, _, source = _fixture(tmp_path)
    draft = {
        "topics": [
            {
                "question": "提交日期？",
                "claims": [
                    {"text": "出差结束后30天内提交报销。", "quotes": [source]},
                    {"text": "开具发票后30天内提交报销。", "quotes": [source]},
                    {"text": "出差结束后3天内提交报销。", "quotes": [source]},
                ],
            },
            {"question": "需要打印发票吗？", "claims": []},
        ]
    }
    responses = {
        "extract": draft,
        "verify": {
            "judgments": [
                {"claim_id": "1", "verdict": "supported", "evidence_ids": ["1.1"]},
                {"claim_id": "2", "verdict": "insufficient", "evidence_ids": ["2.1"]},
                {"claim_id": "3", "verdict": "supported", "evidence_ids": ["3.1"]},
            ]
        },
    }

    class RecordingModel(MockEvidenceModel):
        def verify(self, payload):
            assert [item["claim_id"] for item in payload["claims"]] == ["1", "2", "3"]
            return super().verify(payload)

    service = KnowledgeService(index, RecordingModel(responses))
    answer = service.ask("报销提交日期？需要打印发票吗？")
    assert [item.verdict for item in answer.findings] == [
        "supported",
        "insufficient",
        "insufficient",
        "insufficient",
    ]
    assert "证据不足：需要打印发票吗？" in answer.render()
    assert "行 1–1" in answer.render()
    assert "开具发票后" not in answer.render()
    assert answer.findings[1].text == ""
    assert "待核实候选：开具发票后" in service.ask("报销", mode="annotate").render()


def test_bad_outputs_unknown_ids_and_edits_during_verification_fail_closed(tmp_path):
    index, document, source = _fixture(tmp_path)
    draft = {
        "topics": [
            {"question": "期限？", "claims": [{"text": "期限30天", "quotes": [source]}]}
        ]
    }
    judgment = {"claim_id": "1", "verdict": "supported", "evidence_ids": ["1.1"]}
    for response in (
        {},
        {"judgments": []},
        {"judgments": [judgment, judgment]},
        {"judgments": [{**judgment, "evidence_ids": ["9.1"]}]},
    ):
        answer = KnowledgeService(
            index, MockEvidenceModel({"extract": draft, "verify": response})
        ).ask("报销")
        assert "已核验" not in answer.render()

    class ChangingModel(MockEvidenceModel):
        def verify(self, payload):
            document.write_text("报销期限改为10天。", encoding="utf-8")
            return super().verify(payload)

    answer = KnowledgeService(
        index, ChangingModel({"extract": draft, "verify": {"judgments": [judgment]}})
    ).ask("报销")
    assert answer.findings[0].verdict == "insufficient"
    assert not answer.findings[0].citations


def test_conflict_requires_two_current_sources(tmp_path):
    index, _, first = _fixture(tmp_path)
    quote = "出差结束后15天内提交报销。"
    (tmp_path / "新版.txt").write_text(quote, encoding="utf-8")
    index.refresh()
    second = next(hit for hit in index.search("报销")[1] if hit.path == "新版.txt")
    model = MockEvidenceModel(
        {
            "extract": {
                "topics": [
                    {
                        "question": "期限？",
                        "claims": [
                            {
                                "text": "报销期限为30天",
                                "quotes": [
                                    first,
                                    {"chunk_id": second.chunk_id, "quote": quote},
                                ],
                            }
                        ],
                    }
                ]
            },
            "verify": {
                "judgments": [
                    {
                        "claim_id": "1",
                        "verdict": "conflict",
                        "evidence_ids": ["1.1", "1.2"],
                    }
                ]
            },
        }
    )
    answer = KnowledgeService(index, model).ask("报销")
    assert answer.findings[0].verdict == "conflict"
    assert {cite.hit.path for cite in answer.findings[0].citations} == {
        "差旅报销.txt",
        "新版.txt",
    }


def test_timeout_and_replacing_the_knowledge_base_block_publication(tmp_path):
    index, _, source = _fixture(tmp_path)
    draft = {
        "topics": [
            {"question": "报销", "claims": [{"text": "期限30天", "quotes": [source]}]}
        ]
    }
    response = {
        "extract": draft,
        "verify": {
            "judgments": [
                {"claim_id": "1", "verdict": "supported", "evidence_ids": ["1.1"]}
            ]
        },
    }

    class TimeoutModel(MockEvidenceModel):
        def verify(self, payload):
            raise TimeoutError("private model error")

    class ReplacingModel(MockEvidenceModel):
        def verify(self, payload):
            (tmp_path / "new_docs").mkdir()
            index.refresh("new_docs")
            return super().verify(payload)

    for model, error in [
        (TimeoutModel, "verification_failed"),
        (ReplacingModel, "index_changed"),
    ]:
        answer = KnowledgeService(index, model(response)).ask("报销")
        assert answer.error == error
        assert answer.status == "unverified"
        assert not answer.findings[0].citations
        assert "private model error" not in answer.render()
