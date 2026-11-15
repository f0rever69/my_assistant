from __future__ import annotations

import sys
from types import SimpleNamespace

from my_assistant.config.workspace import Workspace
from my_assistant.sandbox.workspace_search import WorkspaceFTSIndex, _split_terms


def test_chinese_fallback_generates_overlapping_bigrams() -> None:
    words, bigrams = _split_terms("个人助手工作区")

    assert "人助" in bigrams
    assert "手工" in bigrams
    assert words


def test_chinese_tokenizer_uses_jieba_words_when_available(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "jieba",
        SimpleNamespace(cut=lambda _text, cut_all=False: ["个人", "助手"]),
    )

    words, bigrams = _split_terms("个人助手")

    assert words == ["个人", "助手"]
    assert "人助" in bigrams


def test_workspace_search_merges_multiple_query_variants(tmp_path) -> None:
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "chinese.md").write_text(
        "# 本地个人助手\n使用工作区全文索引检索用户资料。", encoding="utf-8"
    )
    (tmp_path / "notes" / "english.md").write_text(
        "# Workspace retrieval\nFull text search helps locate project documents.",
        encoding="utf-8",
    )
    index = WorkspaceFTSIndex(Workspace.from_path(tmp_path))

    hits, stats = index.search(
        "semantic recall",
        ["个人助手 工作区", "workspace retrieval"],
        max_results=8,
    )

    paths = {hit["relative_path"] for hit in hits}
    assert "notes/chinese.md" in paths
    assert "notes/english.md" in paths
    assert stats["documents_in_index"] == 2
    assert all(hit["matched_queries"] for hit in hits)


def test_workspace_search_matches_filename_terms(tmp_path) -> None:
    (tmp_path / "deployment_notes.md").write_text(
        "# Operations\nThe deployment uses a staged rollout.", encoding="utf-8"
    )
    index = WorkspaceFTSIndex(Workspace.from_path(tmp_path))

    hits, _ = index.search("unknown phrase", ["deployment_notes"], max_results=8)

    assert [hit["relative_path"] for hit in hits] == ["deployment_notes.md"]


def test_workspace_search_refreshes_changed_files_and_ignores_sensitive_paths(
    tmp_path,
) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Old phrase\nlegacy material", encoding="utf-8")
    (tmp_path / ".env").write_text("PRIVATE_TOKEN=secret-value", encoding="utf-8")
    index = WorkspaceFTSIndex(Workspace.from_path(tmp_path))

    first, _ = index.search("legacy material", [], max_results=8)
    assert [hit["relative_path"] for hit in first] == ["notes.md"]

    source.write_text("# Current phrase\nupdated project content", encoding="utf-8")
    second, stats = index.search("updated project", [], max_results=8)
    old, _ = index.search("legacy material", [], max_results=8)
    secret, _ = index.search("PRIVATE_TOKEN secret-value", [], max_results=8)

    assert [hit["relative_path"] for hit in second] == ["notes.md"]
    assert old == []
    assert secret == []
    assert stats["files_indexed_or_updated"] == 1


def test_workspace_index_uses_private_runtime_directory(tmp_path) -> None:
    index = WorkspaceFTSIndex(Workspace.from_path(tmp_path))

    assert index.db_path.parent == tmp_path / ".my-assistant" / "index"
    assert index.db_path.exists() is False
