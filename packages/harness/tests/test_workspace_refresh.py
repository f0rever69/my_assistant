"""Three-way host refresh, persistence, and unsafe-path regression coverage."""

import json
import os
from dataclasses import replace

import pytest
from my_assistant.config.workspace import Workspace
from my_assistant.sandbox.file_transactions import FileTransactionConflict
from my_assistant.sandbox.workspace_copy import (
    apply_workspace_copy,
    prepare_workspace_copy,
    refresh_workspace_copy,
    review_workspace_copy,
    workspace_copy_diff,
)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    from my_assistant.config import paths

    monkeypatch.setenv("MY_ASSISTANT_HOME", str(tmp_path / "runtime"))
    monkeypatch.setattr(paths, "_paths", None)
    root = tmp_path / "project"
    root.mkdir()
    (root / "docs").mkdir()
    return replace(Workspace.from_path(root), allowed_read_paths=("docs",))


def put(path, content):
    if content is None:
        path.unlink(missing_ok=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


@pytest.mark.parametrize("base,host,sandbox,status,conflict,preserved", [
    (b"base", b"base", b"base", None, False, False),
    (b"base", b"host", b"base", "modified", False, False),
    (b"base", b"base", b"agent", None, False, True),
    (b"base", b"same", b"same", "aligned", False, False),
    (b"base", b"host", b"agent", None, True, False),
    (None, b"new", None, "added", False, False),
    (None, None, b"agent", None, False, True),
    (None, b"same", b"same", "aligned", False, False),
    (None, b"host", b"agent", None, True, False),
    (b"base", None, b"base", "deleted", False, False),
    (b"base", b"base", None, None, False, True),
    (b"base", None, None, "aligned", False, False),
    (b"base", None, b"agent", None, True, False),
    (b"base", b"host", None, None, True, False),
    (b"", b"\x00\xff\r\n", b"", "modified", False, False),
])
def test_three_way_refresh(workspace, base, host, sandbox, status, conflict, preserved):
    original = workspace.root / "docs/file.txt"
    put(original, base)
    copy = prepare_workspace_copy(workspace, "thread", "default")
    target = copy / "docs/file.txt"
    put(original, host)
    put(target, sandbox)
    baseline_file = copy.parent / "baseline.json"
    before = baseline_file.read_bytes()

    result = refresh_workspace_copy(workspace, "thread", "default")

    assert result["applied"] is not conflict
    assert bool(result["conflicts"]) is conflict
    assert result["preserved"] == (["docs/file.txt"] if preserved else [])
    assert result["changes"] == ([{"path": "docs/file.txt", "status": status}] if status else [])
    expected = sandbox if conflict or preserved else host
    assert (target.read_bytes() if target.is_file() else None) == expected
    assert (original.read_bytes() if original.is_file() else None) == host
    if conflict:
        assert baseline_file.read_bytes() == before
    else:
        again = refresh_workspace_copy(workspace, "thread", "default")
        assert again["applied"] and not again["changes"]
        assert bool(workspace_copy_diff(workspace, "thread", "default")) is preserved


def test_refresh_existing_empty_copy_imports_later_files_and_respects_policy(workspace):
    copy = prepare_workspace_copy(workspace, "old-thread", "default")
    put(workspace.root / "docs/test.txt", b"local file")
    put(workspace.root / "docs/nested/deep/new.txt", b"nested")
    put(workspace.root / "docs/.env", b"private")
    put(workspace.root / "outside.txt", b"outside read scope")
    assert prepare_workspace_copy(workspace, "old-thread", "default") == copy
    assert not (copy / "docs/test.txt").exists()
    identity = copy.stat().st_ino

    result = refresh_workspace_copy(workspace, "old-thread", "default")

    assert result["applied"] and len(result["changes"]) == 2
    assert (copy / "docs/test.txt").read_bytes() == b"local file"
    assert (copy / "docs/nested/deep/new.txt").read_bytes() == b"nested"
    assert not (copy / "docs/.env").exists()
    assert not (copy / "outside.txt").exists()
    assert copy.stat().st_ino == identity


def test_dry_run_and_conflict_preflight_do_not_change_any_files(workspace):
    put(workspace.root / "docs/conflict.txt", b"base")
    copy = prepare_workspace_copy(workspace, "thread", "default")
    put(workspace.root / "docs/added.txt", b"new")
    baseline = (copy.parent / "baseline.json").read_bytes()
    result = refresh_workspace_copy(workspace, "thread", "default", dry_run=True)
    assert not result["applied"] and result["changes"]
    assert not (copy / "docs/added.txt").exists()
    put(workspace.root / "docs/conflict.txt", b"host")
    put(copy / "docs/conflict.txt", b"sandbox")
    result = refresh_workspace_copy(workspace, "thread", "default")
    assert not result["applied"] and result["conflicts"]
    assert not (copy / "docs/added.txt").exists()
    assert (copy.parent / "baseline.json").read_bytes() == baseline


def test_refresh_rebases_reviewed_write_back(workspace):
    put(workspace.root / "docs/file.txt", b"base")
    copy = prepare_workspace_copy(workspace, "thread", "default")
    put(workspace.root / "docs/file.txt", b"new host")
    refresh_workspace_copy(workspace, "thread", "default")
    put(copy / "docs/file.txt", b"agent after refresh")
    review = review_workspace_copy(workspace, "thread", "default", "docs/file.txt")
    apply_workspace_copy(workspace, "thread", "default", "docs/file.txt", review["sha256"])
    assert (workspace.root / "docs/file.txt").read_bytes() == b"agent after refresh"
    assert not workspace_copy_diff(workspace, "thread", "default")


@pytest.mark.parametrize("side", ["host", "sandbox"])
def test_refresh_rejects_link_replacement(workspace, tmp_path, side):
    put(workspace.root / "docs/file.txt", b"base")
    copy = prepare_workspace_copy(workspace, "thread", "default")
    put(workspace.root / "docs/file.txt", b"new host")
    outside = tmp_path / "outside.txt"
    put(outside, b"outside")
    target = (workspace.root if side == "host" else copy) / "docs/file.txt"
    target.unlink()
    try:
        target.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation unavailable")
    baseline = (copy.parent / "baseline.json").read_bytes()
    result = refresh_workspace_copy(workspace, "thread", "default")
    assert not result["applied"] and result["conflicts"]
    assert outside.read_bytes() == b"outside"
    assert (copy.parent / "baseline.json").read_bytes() == baseline


def test_refresh_rejects_hardlink_destination(workspace, tmp_path):
    copy = prepare_workspace_copy(workspace, "thread", "default")
    put(workspace.root / "docs/file.txt", b"new host")
    outside = tmp_path / "outside.txt"
    put(outside, b"outside")
    (copy / "docs").mkdir(exist_ok=True)
    try:
        os.link(outside, copy / "docs/file.txt")
    except OSError:
        pytest.skip("hardlink creation unavailable")
    result = refresh_workspace_copy(workspace, "thread", "default")
    assert not result["applied"] and result["conflicts"]
    assert outside.read_bytes() == b"outside"


@pytest.mark.parametrize("directory_in_copy", [False, True])
def test_refresh_rejects_file_directory_collisions(workspace, directory_in_copy):
    copy = prepare_workspace_copy(workspace, "thread", "default")
    if directory_in_copy:
        put(copy / "docs/new/agent.txt", b"keep")
        put(workspace.root / "docs/new", b"host file")
    else:
        put(copy / "docs/new", b"keep")
        put(workspace.root / "docs/new/host.txt", b"host file")
    result = refresh_workspace_copy(workspace, "thread", "default")
    assert not result["applied"] and result["conflicts"]
    target = copy / ("docs/new/agent.txt" if directory_in_copy else "docs/new")
    assert target.read_bytes() == b"keep"


@pytest.mark.parametrize("name", ["../escape", "/absolute", "docs/../escape", "docs/.env", "C:/escape"])
def test_refresh_rejects_malformed_or_protected_baseline(workspace, name):
    copy = prepare_workspace_copy(workspace, "thread", "default")
    (copy.parent / "baseline.json").write_text(json.dumps({name: "0" * 64}), encoding="utf-8")
    with pytest.raises((ValueError, PermissionError)):
        refresh_workspace_copy(workspace, "thread", "default")


@pytest.mark.parametrize("side", ["host", "sandbox"])
def test_refresh_detects_changes_after_preflight(workspace, monkeypatch, side):
    from my_assistant.sandbox import workspace_copy as module

    put(workspace.root / "docs/file.txt", b"base")
    copy = prepare_workspace_copy(workspace, "thread", "default")
    put(workspace.root / "docs/file.txt", b"new host")
    original = module._check_refresh_versions
    calls = 0

    def race(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            put((workspace.root if side == "host" else copy) / "docs/file.txt", b"concurrent")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_check_refresh_versions", race)
    with pytest.raises(FileTransactionConflict, match="during refresh"):
        refresh_workspace_copy(workspace, "thread", "default")
    assert (copy / "docs/file.txt").read_bytes() == (b"concurrent" if side == "sandbox" else b"base")
    assert json.loads((copy.parent / "baseline.json").read_text())["docs/file.txt"] != module.hash_bytes(b"new host")


def test_refresh_is_scoped_to_user_and_thread(workspace):
    copy = prepare_workspace_copy(workspace, "thread", "alice")
    other = prepare_workspace_copy(workspace, "thread", "bob")
    another = prepare_workspace_copy(workspace, "another", "alice")
    put(workspace.root / "docs/new.txt", b"host")
    refresh_workspace_copy(workspace, "thread", "alice")
    assert (copy / "docs/new.txt").exists()
    assert not (other / "docs/new.txt").exists()
    assert not (another / "docs/new.txt").exists()


def test_refresh_requires_existing_copy(workspace):
    with pytest.raises(FileNotFoundError, match="no Docker Workspace copy"):
        refresh_workspace_copy(workspace, "missing", "default")


def test_interrupted_refresh_can_be_retried(workspace, monkeypatch):
    from my_assistant.sandbox import workspace_copy as module

    copy = prepare_workspace_copy(workspace, "thread", "default")
    put(workspace.root / "docs/a.txt", b"a")
    put(workspace.root / "docs/b.txt", b"b")
    writer = module._write_refresh_file

    def interrupt(staged, name, *args):
        if name == "docs/b.txt":
            raise OSError("simulated disk error")
        return writer(staged, name, *args)

    monkeypatch.setattr(module, "_write_refresh_file", interrupt)
    with pytest.raises(OSError, match="disk error"):
        refresh_workspace_copy(workspace, "thread", "default")
    assert (copy / "docs/a.txt").read_bytes() == b"a"
    assert not (copy / "docs/b.txt").exists()
    assert json.loads((copy.parent / "baseline.json").read_text()) == {}
    monkeypatch.setattr(module, "_write_refresh_file", writer)
    result = refresh_workspace_copy(workspace, "thread", "default")
    assert result["applied"] and not result["conflicts"]
    assert (copy / "docs/b.txt").read_bytes() == b"b"
    assert not workspace_copy_diff(workspace, "thread", "default")


def test_refresh_preserves_source_file_permissions(workspace):
    if os.name == "nt":
        pytest.skip("POSIX executable mode is not represented on Windows")
    copy = prepare_workspace_copy(workspace, "thread", "default")
    source = workspace.root / "docs/script.sh"
    put(source, b"#!/bin/sh\necho test\n")
    source.chmod(0o755)
    refresh_workspace_copy(workspace, "thread", "default")
    assert (copy / "docs/script.sh").stat().st_mode & 0o777 == 0o755


@pytest.mark.parametrize("side", ["host", "sandbox"])
def test_refresh_rejects_reparse_points(workspace, monkeypatch, side):
    from my_assistant.sandbox import workspace_copy as module

    put(workspace.root / "docs/file.txt", b"base")
    copy = prepare_workspace_copy(workspace, "thread", "default")
    put(workspace.root / "docs/file.txt", b"host")
    target = (workspace.root if side == "host" else copy) / "docs/file.txt"
    inode = target.lstat().st_ino
    original = module._is_reparse_point
    monkeypatch.setattr(module, "_is_reparse_point", lambda info: info.st_ino == inode or original(info))
    result = refresh_workspace_copy(workspace, "thread", "default")
    assert not result["applied"] and result["conflicts"]
    assert (copy / "docs/file.txt").read_bytes() == b"base"


@pytest.mark.parametrize("limit", ["_MAX_FILE_BYTES", "_MAX_COPY_BYTES"])
def test_refresh_enforces_copy_size_limits(workspace, monkeypatch, limit):
    from my_assistant.sandbox import workspace_copy as module

    copy = prepare_workspace_copy(workspace, "thread", "default")
    put(workspace.root / "docs/a.txt", b"123")
    put(workspace.root / "docs/b.txt", b"456")
    monkeypatch.setattr(module, limit, 2 if limit == "_MAX_FILE_BYTES" else 5)
    with pytest.raises(ValueError, match="exceeds"):
        refresh_workspace_copy(workspace, "thread", "default")
    assert list(copy.iterdir()) == []


@pytest.mark.parametrize("baseline", [[], {"docs/x": None}, {"docs/x": "bad hash"}])
def test_refresh_rejects_corrupted_baseline(workspace, baseline):
    copy = prepare_workspace_copy(workspace, "thread", "default")
    (copy.parent / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        refresh_workspace_copy(workspace, "thread", "default")
