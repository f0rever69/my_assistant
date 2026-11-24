import asyncio
import json
import os
import subprocess
import uuid
from dataclasses import replace

import pytest
from my_assistant.config.workspace import Workspace, workspace_scope
from my_assistant.sandbox.file_transactions import FileTransactionConflict
from my_assistant.sandbox.workspace_copy import (
    apply_workspace_copy,
    prepare_workspace_copy,
    review_workspace_copy,
    workspace_copy_diff,
)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    from my_assistant.config import paths

    monkeypatch.setenv("MY_ASSISTANT_HOME", str(tmp_path / "runtime"))
    monkeypatch.setattr(paths, "_paths", None)
    project = tmp_path / "project"
    project.mkdir()
    (project / "docs").mkdir()
    (project / "docs" / "sample.txt").write_text("project source", encoding="utf-8")
    (project / "docs" / ".env").write_text("private fixture", encoding="utf-8")
    (project / "private.txt").write_text("outside read scope", encoding="utf-8")
    return replace(
        Workspace.from_path(project),
        allowed_read_paths=("docs",),
        allowed_write_paths=("docs",),
        shell_enabled=True,
    )


def test_filtered_copy_and_reviewed_write_back(workspace):
    owner, thread = "default", "copy-check"
    copy = prepare_workspace_copy(workspace, thread, owner)
    assert sorted(p.name for p in (copy / "docs").iterdir()) == ["sample.txt"]
    result = copy / "docs" / "sample.txt"
    original = workspace.root / "docs" / "sample.txt"
    result.write_text("reviewed result", encoding="utf-8")
    assert original.read_text(encoding="utf-8") == "project source"
    review = review_workspace_copy(workspace, thread, owner, "docs/sample.txt")
    assert "+reviewed result" in review["diff"]
    assert (
        workspace_copy_diff(workspace, thread, owner)[0]["sha256"] == review["sha256"]
    )
    result.write_text("revised after review", encoding="utf-8")
    with pytest.raises(FileTransactionConflict, match="since review"):
        apply_workspace_copy(
            workspace, thread, owner, "docs/sample.txt", review["sha256"]
        )
    result.write_text("reviewed result", encoding="utf-8")
    original.write_text("new host version", encoding="utf-8")
    with pytest.raises(FileTransactionConflict, match="Host file changed"):
        apply_workspace_copy(
            workspace, thread, owner, "docs/sample.txt", review["sha256"]
        )
    original.write_text("project source", encoding="utf-8")
    apply_workspace_copy(workspace, thread, owner, "docs/sample.txt", review["sha256"])
    assert original.read_text(encoding="utf-8") == "reviewed result"
    assert not workspace_copy_diff(workspace, thread, owner)
    (copy / "outside.txt").write_text("blocked", encoding="utf-8")
    with pytest.raises(PermissionError):
        review_workspace_copy(workspace, thread, owner, "outside.txt")
    assert not (
        prepare_workspace_copy(workspace, "other-thread", owner) / "outside.txt"
    ).exists()


@pytest.mark.skipif(
    os.environ.get("MY_ASSISTANT_TEST_DOCKER") != "1",
    reason="requires local Docker and AIO image",
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_acquire_recovers_leftover_networks(workspace, monkeypatch, asynchronous):
    from my_assistant.community.aio_sandbox import AioSandboxProvider
    from my_assistant.config import get_app_config

    config = get_app_config()
    monkeypatch.setattr(config.sandbox, "container_prefix", f"ma-recovery-{uuid.uuid4().hex[:10]}")
    provider = AioSandboxProvider()
    thread = "orphan-recovery"
    with workspace_scope(workspace):
        sandbox_id = provider._sandbox_id_for_thread(thread, "default")
    backend = provider._backend
    _, network = backend._resource_names(sandbox_id)
    egress = backend._egress_network_name(sandbox_id)

    async def acquire_async():
        with workspace_scope(workspace):
            return await provider.acquire_async(thread, user_id="default")

    try:
        backend._create_internal_network(network, sandbox_id)
        backend._create_egress_network(egress, sandbox_id)
        discovered = backend.discover(sandbox_id)
        assert discovered is not None and discovered.requires_replacement
        if asynchronous:
            acquired = asyncio.run(acquire_async())
        else:
            with workspace_scope(workspace):
                acquired = provider.acquire(thread, user_id="default")
        assert acquired == sandbox_id
        assert provider.get(acquired).read_file("/mnt/user-data/workspace/docs/sample.txt").strip() == "project source"
        assert backend._restricted_resources_status(sandbox_id) == "compatible"
        provider.release(acquired)
        assert backend._restricted_resources_status(sandbox_id) == "missing"
    finally:
        provider.shutdown()
        backend._cleanup_restricted_resources(sandbox_id)


@pytest.mark.skipif(
    os.environ.get("MY_ASSISTANT_TEST_DOCKER") != "1",
    reason="requires local Docker and AIO image",
)
def test_real_docker_isolation_and_timeout(workspace):
    from my_assistant.community.aio_sandbox import AioSandboxProvider

    provider = AioSandboxProvider()
    assert provider.sandbox_network_mode() == "isolated"
    assert provider._config["isolate_workspace"]
    owner, thread = "default", "real-docker"
    paths = workspace.ensure_runtime_dirs(thread)
    upload = workspace.resolve(paths["uploads_path"]) / "input.txt"
    upload.write_text("uploaded file", encoding="utf-8")

    async def acquire():
        with workspace_scope(workspace):
            return await provider.acquire_async(thread, user_id=owner)

    try:
        sandbox_id = asyncio.run(acquire())
        sandbox = provider.get(sandbox_id)
        assert sandbox is not None
        info = provider._sandbox_infos[sandbox_id]
        inspected = subprocess.run(
            ["docker", "container", "inspect", info.container_id],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        [data] = json.loads(inspected.stdout)
        mounts = {mount["Destination"]: mount for mount in data["Mounts"]}
        assert not mounts["/mnt/user-data/uploads"]["RW"]
        assert mounts["/mnt/user-data/workspace"]["RW"]
        assert not any(
            mount["Destination"] in {"/var/run/docker.sock", "/mnt/acp-workspace"}
            for mount in mounts.values()
        )
        assert "seccomp=unconfined" not in data["HostConfig"]["SecurityOpt"]
        assert not data["HostConfig"]["Privileged"]
        assert data["HostConfig"]["PidsLimit"] > 0 and data["HostConfig"]["Memory"] > 0
        root = "/mnt/user-data/workspace"
        assert sandbox.read_file(f"{root}/docs/sample.txt").strip() == "project source"
        assert (
            sandbox.execute_command(
                f"test ! -e {root}/docs/.env && test ! -e {root}/private.txt && echo filtered"
            ).strip()
            == "filtered"
        )
        assert (
            sandbox.read_file("/mnt/user-data/uploads/input.txt").strip()
            == "uploaded file"
        )
        read_only = sandbox.execute_command(
            "echo rejected >> /mnt/user-data/uploads/input.txt"
        )
        assert "Exit Code:" in read_only
        assert upload.read_text(encoding="utf-8") == "uploaded file"
        assert (
            sandbox.execute_command(
                'test "$(id -u)" -ne 0 && echo unprivileged'
            ).strip()
            == "unprivileged"
        )
        proxy_result = sandbox.execute_command(
            "curl -sS -o /dev/null -w '%{http_code}' --max-time 5 http://example.com"
        )
        assert "403" in proxy_result
        direct_result = sandbox.execute_command(
            "curl --noproxy '*' -sS --max-time 3 https://1.1.1.1"
        )
        assert "Exit Code:" in direct_result
        sandbox.write_file(f"{root}/docs/sample.txt", "container result")
        sandbox.write_file("/mnt/user-data/outputs/result.txt", "thread output")
        assert (workspace.resolve(paths["outputs_path"]) / "result.txt").read_text(
            encoding="utf-8"
        ).strip() == "thread output"
        assert (workspace.root / "docs" / "sample.txt").read_text(
            encoding="utf-8"
        ) == "project source"
        ended = sandbox.execute_command("sleep 45 & wait", timeout=1)
        assert "Docker sandbox terminated" in ended
        inspect_stopped = subprocess.run(
            [
                "docker",
                "container",
                "inspect",
                "--format",
                "{{.State.Running}}",
                info.container_id,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        assert (
            inspect_stopped.returncode != 0 or inspect_stopped.stdout.strip() == "false"
        )
        provider.release(sandbox_id)
        assert provider.get(sandbox_id) is None
        review = review_workspace_copy(workspace, thread, owner, "docs/sample.txt")
        apply_workspace_copy(
            workspace, thread, owner, "docs/sample.txt", review["sha256"]
        )
        assert (workspace.root / "docs" / "sample.txt").read_text(
            encoding="utf-8"
        ).strip() == "container result"
    finally:
        provider.shutdown()


@pytest.mark.skipif(
    os.environ.get("MY_ASSISTANT_TEST_DOCKER") != "1",
    reason="requires local Docker and AIO image",
)
def test_local_cli_uses_docker_workspace(workspace, monkeypatch, capsys):
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage, ToolMessage
    from my_assistant import client as client_module
    from my_assistant.config import get_app_config
    from my_assistant.runtime.checkpointer.provider import reset_checkpointer
    from my_assistant.sandbox.sandbox_provider import get_sandbox_provider
    from my_assistant.tui.cli import main
    from my_assistant.tui.session import open_session
    from my_assistant.workspaces import get_workspace_registry

    class ScriptedModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            assert {"bash", "present_files"} <= {tool.name for tool in tools}
            assert not {"web_search", "web_fetch", "invoke_acp_agent"} & {
                tool.name for tool in tools
            }
            return self

        def _generate(self, messages, **kwargs):
            tool_messages = [
                message for message in messages if isinstance(message, ToolMessage)
            ]
            if tool_messages:
                latest = tool_messages[-1]
                if latest.tool_call_id == "bash-call":
                    assert "project source" in latest.content
                    assert "Exit Code:" not in latest.content
                else:
                    assert latest.tool_call_id == "present-call"
                    assert latest.content == "Successfully presented files"
            return super()._generate(messages, **kwargs)

    def create_model(*args, **kwargs):
        return ScriptedModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "bash",
                            "id": "bash-call",
                            "args": {
                                "command": (
                                    "cat docs/sample.txt && printf 'cli result' > docs/cli.txt "
                                    "&& printf 'output artifact' > /mnt/user-data/outputs/cli.txt"
                                )
                            },
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "present_files",
                            "id": "present-call",
                            "args": {"filepaths": ["/mnt/user-data/outputs/cli.txt"]},
                        }
                    ],
                ),
                AIMessage(content="Docker CLI verified"),
            ]
        )

    config = get_app_config()
    monkeypatch.setattr(config.memory, "enabled", False)
    monkeypatch.setattr(config.title, "enabled", False)
    monkeypatch.setattr(config.database, "sqlite_dir", str(workspace.root.parent / "db"))
    monkeypatch.setattr(config.run_events, "backend", "memory")
    monkeypatch.setattr(client_module, "create_chat_model", create_model)
    reset_checkpointer()
    registry = get_workspace_registry()
    registered = registry.register(workspace.root)
    registered = registry.update(
        registered.id,
        shell_enabled=True,
        allowed_read_paths=["docs"],
        allowed_write_paths=["docs"],
    )
    provider = get_sandbox_provider()
    thread = "docker-cli-review"
    try:
        assert (
            main(
                [
                    "--print",
                    "Read the project file and create a result",
                    "--workspace-id",
                    registered.id,
                    "--resume",
                    thread,
                    "--recursion-limit",
                    "120",
                ]
            )
            == 0
        )
        assert "Docker CLI verified" in capsys.readouterr().out
        assert not (workspace.root / "docs" / "cli.txt").exists()
        output = workspace.runtime_dir(thread) / "outputs" / "cli.txt"
        assert output.read_text(encoding="utf-8") == "output artifact"
        assert not provider._sandboxes
        assert not provider._warm_pool
        session = open_session(persistence=False)
        try:
            context = registered.to_context()
            changes = session.client.diff_workspace_copy(thread, context)
            assert [change["path"] for change in changes] == ["docs/cli.txt"]
            reviewed = session.client.review_workspace_copy(
                thread, context, "docs/cli.txt"
            )
            assert "+cli result" in reviewed["diff"]
            session.client.apply_workspace_copy(
                thread, context, "docs/cli.txt", reviewed["sha256"]
            )
            assert (workspace.root / "docs" / "cli.txt").read_text(
                encoding="utf-8"
            ) == "cli result"
        finally:
            session.close()
    finally:
        provider.shutdown()
