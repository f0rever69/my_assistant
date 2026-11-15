from __future__ import annotations

from pathlib import Path

import pytest

from my_assistant.config.workspace import Workspace
from my_assistant.tui.cli import plan_launch
from my_assistant.tui.render import render_header
from my_assistant.workspace_changes.scanner import is_sensitive_workspace_path


def test_workspace_rejects_paths_outside_root_and_ignored_files(tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    (root / ".env").write_text("TOKEN=secret", encoding="utf-8")
    (root / ".env.local").write_text("TOKEN=secret", encoding="utf-8")
    workspace = Workspace.from_path(root)

    assert workspace.check("README.md") == root / "README.md"
    for path in ("../outside.txt", ".env", ".env.local", ".my-assistant/threads/x/checkpoint.db"):
        with pytest.raises(PermissionError):
            workspace.check(path)


def test_workspace_rejects_symlink_escape(tmp_path) -> None:
    root = tmp_path / "project"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(PermissionError):
        Workspace.from_path(root).check(link / "secret.txt")


def test_workspace_rejects_symlink_even_when_target_stays_inside(tmp_path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    real = root / "real"
    real.mkdir()
    (real / "note.txt").write_text("safe", encoding="utf-8")
    link = root / "alias"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")

    workspace = Workspace.from_path(root)
    with pytest.raises(PermissionError, match="symlinks or reparse points"):
        workspace.check("alias/note.txt")


def test_workspace_rejects_symlinked_parent_for_new_write(tmp_path) -> None:
    root = tmp_path / "workspace"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")

    workspace = Workspace.from_path(root)
    with pytest.raises(PermissionError, match="symlinks or reparse points"):
        workspace.check("linked/new.txt", write=True)


@pytest.mark.parametrize(
    "path",
    [
        "/mnt/user-data/workspace/.env",
        "/mnt/user-data/workspace/.ssh/config",
        "/mnt/user-data/workspace/.ssh/id_ed25519",
        "/mnt/user-data/workspace/authorized_keys",
        "/mnt/user-data/workspace/service.credentials.json",
        "/mnt/user-data/workspace/client.pem",
    ],
)
def test_sensitive_workspace_paths_are_not_included_in_text_diffs(path) -> None:
    assert is_sensitive_workspace_path(path)


def test_workspace_outputs_are_agent_visible_but_private_runtime_is_not(tmp_path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    workspace = Workspace.from_path(root)
    dirs = workspace.ensure_runtime_dirs("thread-1")

    assert workspace.check(Path(dirs["outputs_path"]) / "result.txt", write=True)
    with pytest.raises(PermissionError):
        workspace.check(workspace.runtime_dir("thread-1") / "checkpoint.json")


def test_workspace_cli_defaults_capabilities_to_disabled(tmp_path) -> None:
    plan = plan_launch(
        ["--workspace", str(tmp_path), "--print", "inspect files"],
        stdin_isatty=True,
        stdout_isatty=True,
        env={},
    )

    assert plan.mode == "print"
    assert plan.workspace_path == str(tmp_path)
    assert plan.workspace_shell is False
    assert plan.workspace_network is False


def test_workspace_id_launch_plan_and_header_expose_policy(tmp_path) -> None:
    plan = plan_launch(
        ["--workspace-id", "ws-abc", "--print", "inspect"],
        stdin_isatty=True,
        stdout_isatty=True,
        env={},
    )

    assert plan.workspace_id == "ws-abc"
    assert plan.workspace_path is None
    root = tmp_path / "project"
    root.mkdir()
    header = render_header(
        model="test-model",
        thread_label="thread abc",
        cwd="C:/cwd",
        skills=2,
        workspace={
            "id": "ws-12345678",
            "name": "Project",
            "root_path": str(root),
            "allowed_read_paths": ["."],
            "allowed_write_paths": ["src"],
            "shell_enabled": False,
            "network_enabled": True,
        },
    )
    rendered = header.plain
    assert "Project" in rendered
    assert "R1/W1" in rendered
    assert "shell:off" in rendered and "net:on" in rendered
    assert str(root) in rendered
