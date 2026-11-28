"""Review long files and version-pinned changes without writing the originals."""

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest
from my_assistant.config.workspace import Workspace
from my_assistant.sandbox.file_transactions import FileTransactionConflict
from my_assistant.sandbox.workspace_copy import prepare_workspace_copy
from my_assistant.sandbox.workspace_review import (
    apply_workspace_review,
    compare_workspace_file,
    list_workspace_review,
)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_ASSISTANT_HOME", str(tmp_path / "state"))
    root = tmp_path / "project"
    root.mkdir()
    return Workspace.from_path(root)


def test_large_text_compare_exports_exact_versions_without_terminal_truncation(project):
    name = "长文稿.md"
    original = b"Original paragraph.\n" * 20_000
    candidate = b"Revised paragraph.\n" * 22_000
    path = project.root / name
    path.write_bytes(original)
    staged = prepare_workspace_copy(project, "review", "default") / name
    staged.write_bytes(candidate)

    compared = compare_workspace_file(project, "review", "default", name)
    assert "超出终端预览范围" in compared.text_preview
    assert compared.is_text and compared.original.read_bytes() == original
    assert compared.candidate.read_bytes() == candidate and path.read_bytes() == original
    assert not compared.candidate.is_relative_to(staged.parent)
    assert compare_workspace_file(project, "review", "default", name).candidate == compared.candidate


def _docx(text: str) -> bytes:
    content = BytesIO()
    with ZipFile(content, mode="w") as archive:
        archive.writestr("word/document.xml", f"""
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body>
            </w:document>
        """)
    return content.getvalue()


def test_docx_compare_shows_paragraph_changes_and_keeps_full_files(project):
    name = "report.docx"
    original, candidate = _docx("原始段落"), _docx("改写段落")
    (project.root / name).write_bytes(original)
    copy = prepare_workspace_copy(project, "review", "default")
    (copy / name).write_bytes(candidate)

    compared = compare_workspace_file(project, "review", "default", name)
    assert "-原始段落\n+改写段落" in compared.text_preview
    assert not compared.is_text
    assert compared.original.read_bytes() == original
    assert compared.candidate.read_bytes() == candidate


def test_batch_writes_selected_add_modify_delete_and_keeps_backups(project):
    for name in ("modified.txt", "deleted.txt", "unselected.txt", "conflict.txt"):
        (project.root / name).write_bytes(b"base")
    copy = prepare_workspace_copy(project, "review", "default")
    for name in ("added.txt", "modified.txt", "unselected.txt", "conflict.txt"):
        (copy / name).write_bytes(b"new")
    (copy / "deleted.txt").unlink()
    (project.root / "conflict.txt").write_bytes(b"local")

    changes = {item.path: item for item in list_workspace_review(project, "review", "default")}
    conflict = changes["conflict.txt"]
    assert conflict.blocked_reason
    with pytest.raises(FileTransactionConflict, match="不是可批量写回"):
        apply_workspace_review(project, "review", "default", {conflict.path: conflict.token})
    allowed = ("added.txt", "modified.txt", "deleted.txt")
    result = apply_workspace_review(project, "review", "default", {name: changes[name].token for name in allowed})

    assert result["error"] is None
    assert [item["path"] for item in result["applied"]] == list(allowed)
    assert (project.root / "added.txt").read_bytes() == (project.root / "modified.txt").read_bytes() == b"new"
    assert not (project.root / "deleted.txt").exists()
    assert (project.root / "unselected.txt").read_bytes() == b"base"
    assert (project.root / "conflict.txt").read_bytes() == b"local"
    backup = Path(result["applied"][-1]["backup_path"])
    assert (backup / "local.bin").read_bytes() == b"base"
    assert {item.path for item in list_workspace_review(project, "review", "default")} == {"unselected.txt", "conflict.txt"}


@pytest.mark.parametrize("changed_side", ["original", "candidate"])
def test_batch_preflight_never_writes_earlier_file_when_later_version_changed(project, changed_side):
    (project.root / "b.txt").write_bytes(b"base")
    copy = prepare_workspace_copy(project, "review", "default")
    (copy / "a.txt").write_bytes(b"new a")
    (copy / "b.txt").write_bytes(b"new b")
    selection = {item.path: item.token for item in list_workspace_review(project, "review", "default")}
    changed = project.root if changed_side == "original" else copy
    (changed / "b.txt").write_bytes(b"late edit")

    with pytest.raises(FileTransactionConflict, match="版本已变化"):
        apply_workspace_review(project, "review", "default", selection)
    assert not (project.root / "a.txt").exists()
    with pytest.raises(FileTransactionConflict, match="版本已变化"):
        compare_workspace_file(project, "review", "default", "b.txt", expected_token=selection["b.txt"])


def test_batch_reports_completed_files_and_stops_after_write_error(project, monkeypatch):
    from my_assistant.sandbox import workspace_review

    copy = prepare_workspace_copy(project, "review", "default")
    for name in ("a.txt", "b.txt", "c.txt"):
        (copy / name).write_bytes(b"new")
    selection = {item.path: item.token for item in list_workspace_review(project, "review", "default")}
    original_apply = workspace_review._apply_reviewed_version_locked

    def fail_second(workspace, root, thread_id, user_id, path, *args, **kwargs):
        if path == "b.txt":
            raise OSError("disk unavailable")
        return original_apply(workspace, root, thread_id, user_id, path, *args, **kwargs)

    monkeypatch.setattr(workspace_review, "_apply_reviewed_version_locked", fail_second)
    result = apply_workspace_review(project, "review", "default", selection)
    assert [item["path"] for item in result["applied"]] == ["a.txt"]
    assert result["failed_path"] == "b.txt" and result["not_attempted"] == ["c.txt"]
    assert [file.name for file in project.root.glob("*.txt")] == ["a.txt"]


def test_workspace_copy_retries_temporary_windows_rename_lock(project, monkeypatch):
    original_rename = Path.rename
    calls = []

    def rename_once_locked(path, destination):
        if path.name == "copy" and path.parent.name.startswith("stage-"):
            calls.append(destination)
            if len(calls) == 1:
                error = PermissionError("transient file lock")
                error.winerror = 5
                raise error
        return original_rename(path, destination)

    (project.root / "note.txt").write_text("before")
    monkeypatch.setattr(Path, "rename", rename_once_locked)
    staged = prepare_workspace_copy(project, "review", "default")
    assert len(calls) == 2 and calls[0] == calls[1]
    assert (staged / "note.txt").read_text() == "before"
