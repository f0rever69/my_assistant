from dataclasses import replace

import pytest
from my_assistant.config.workspace import Workspace
from my_assistant.rag.index import KnowledgeIndex


def test_chinese_bm25_returns_passages_and_reuses_index(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "制度.md").write_text(
        "# 差旅报销\n\n出差结束后 30 天内提交报销。\n\n办公设备需要登记。",
        encoding="utf-8",
    )
    (tmp_path / "other.txt").write_text("差旅报销", encoding="utf-8")
    index = KnowledgeIndex(tmp_path / "kb.sqlite", Workspace.from_path(tmp_path))
    with pytest.raises(ValueError, match="Build"):
        index.search("差旅报销")
    assert index.refresh("docs")["indexed"] == 1
    revision, hits = index.search("报销期限", variants=("出差 提交",))
    assert revision == 1
    assert hits[0].path == "docs/制度.md"
    assert hits[0].location == "行 3–3"
    assert "30 天" in hits[0].text
    assert index.refresh()["imported"] == 0
    assert (
        index.search("报销期限", variants=("出差 提交",))[1][0].chunk_id
        == hits[0].chunk_id
    )


def test_changed_deleted_and_revoked_sources_are_not_returned(tmp_path):
    document = tmp_path / "travel.txt"
    document.write_text("差旅报销期限为30天", encoding="utf-8")
    workspace = Workspace.from_path(tmp_path)
    active = workspace
    index = KnowledgeIndex(
        tmp_path / "kb.sqlite", workspace, resolve_workspace=lambda: active
    )
    index.refresh(".")
    original = index.search("差旅报销")[1][0]
    document.write_text("差旅报销期限为15天", encoding="utf-8")
    assert index.search("差旅报销")[1] == []
    index.refresh()
    assert index.search("差旅报销")[1][0].version != original.version
    active = replace(workspace, allowed_read_paths=())
    assert index.search("差旅报销")[1] == []
    active = workspace
    document.unlink()
    assert index.refresh()["indexed"] == 0
    assert index.list_documents() == []


def test_refresh_switches_source_and_reports_failed_import(tmp_path):
    for name in ("one", "two"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "制度.txt").write_text("出差补贴", encoding="utf-8")
    (tmp_path / "two" / "bad.pdf").write_text("bad pdf", encoding="utf-8")
    index = KnowledgeIndex(tmp_path / "kb.sqlite", Workspace.from_path(tmp_path))
    index.refresh("one")
    assert index.refresh("two")["skipped"] == [
        {"path": "two/bad.pdf", "reason": "extraction_failed"}
    ]
    assert [hit.path for hit in index.search("出差补贴")[1]] == ["two/制度.txt"]
