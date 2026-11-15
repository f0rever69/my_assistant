from dataclasses import replace

from my_assistant.config.workspace import Workspace
from my_assistant.rag.evidence import CitationVerifier, QuoteRequest, numbers_supported
from my_assistant.rag.index import KnowledgeIndex


def test_citation_is_exact_and_bound_to_current_workspace_source(tmp_path):
    document = tmp_path / "制度.txt"
    document.write_text("报销期限为３０天。", encoding="utf-8")
    workspace = Workspace.from_path(tmp_path)
    active = workspace
    index = KnowledgeIndex(
        tmp_path / "kb.sqlite", workspace, resolve_workspace=lambda: active
    )
    index.refresh(".")
    [hit] = index.search("报销期限")[1]
    request = QuoteRequest(chunk_id=hit.chunk_id, quote="报销期限为３０天。")
    citation, status = CitationVerifier(index, [hit]).verify(request)
    assert status == "verified"
    assert citation.quote == request.quote
    assert numbers_supported("30天", [citation])
    assert not numbers_supported("3天", [citation])
    assert not numbers_supported("30个小时", [citation])
    assert not numbers_supported("30工作日", [citation])
    assert CitationVerifier(index, []).verify(request)[1] == "unknown_chunk"
    assert (
        CitationVerifier(index, [hit]).verify(
            request.model_copy(update={"quote": "30天"})
        )[1]
        == "quote_mismatch"
    )
    assert (
        CitationVerifier(index, [replace(hit, location="行 9")]).verify(request)[1]
        == "source_mismatch"
    )
    active = replace(workspace, allowed_read_paths=())
    assert CitationVerifier(index, [hit]).verify(request)[1] == "source_unavailable"
    active = workspace
    document.write_text("报销期限为１５天。", encoding="utf-8")
    assert CitationVerifier(index, [hit]).verify(request)[1] == "stale_source"
