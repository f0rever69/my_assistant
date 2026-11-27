"""Location evidence, misleading model outputs, and explicit conflict resolution."""

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from my_assistant.agents.middlewares.file_delivery_middleware import (
    FileEffectMiddleware,
    WorkspaceDeliveryMiddleware,
)
from my_assistant.agents.middlewares.tool_receipt import (
    TOOL_RECEIPT_KEY,
    extract_citing_turn_receipts,
    extract_tool_receipts,
    is_valid_receipt,
    make_tool_receipt,
    render_tool_receipts_with_snapshot,
)
from my_assistant.agents.middlewares.tool_receipt_middleware import (
    ToolReceiptMiddleware,
)
from my_assistant.config.workspace import Workspace
from my_assistant.sandbox.file_effects import FILE_EFFECT_KEY
from my_assistant.sandbox.file_transactions import FileTransactionConflict
from my_assistant.sandbox.tools import bash_tool, str_replace_tool, write_file_tool
from my_assistant.sandbox.workspace_copy import (
    apply_workspace_copy,
    prepare_workspace_copy,
    resolve_workspace_conflict,
    review_workspace_conflict,
    review_workspace_copy,
    workspace_copy_status,
)


@pytest.fixture
def scene(tmp_path, monkeypatch):
    from my_assistant.config import paths

    monkeypatch.setenv("MY_ASSISTANT_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(paths, "_paths", None)
    root = tmp_path / "project"
    root.mkdir()
    (root / "test.txt").write_text("original", encoding="utf-8")
    workspace = Workspace.from_path(root)
    copy = prepare_workspace_copy(workspace, "thread", "default")
    state = {"messages": [HumanMessage("edit test.txt")], "sandbox": {"sandbox_id": "aio-id"}, "thread_data": {"workspace_policy": workspace.to_context()}}
    runtime = SimpleNamespace(state=state, context={"thread_id": "thread", "workspace": workspace.to_context(), "user_id": "default"})
    config = SimpleNamespace(sandbox=SimpleNamespace(use="my_assistant.community.aio_sandbox:AioSandboxProvider", isolate_workspace=True))
    return workspace, copy, runtime, config


def request_for(runtime, native=write_file_tool):
    return SimpleNamespace(runtime=runtime, state=runtime.state, tool=native, tool_call={"id": "call", "name": native.name, "args": {"path": "/mnt/user-data/workspace/test.txt"}})


@pytest.mark.parametrize("native", [write_file_tool, str_replace_tool, bash_tool])
def test_tool_receipts_include_location_and_survive_ledger_round_trip(scene, native):
    _, _, runtime, config = scene
    request = request_for(runtime, native)
    effects = FileEffectMiddleware(config)
    result = ToolReceiptMiddleware().wrap_tool_call(request, lambda req: effects.wrap_tool_call(req, lambda _: ToolMessage("OK", tool_call_id="call")))
    effect = result.additional_kwargs[FILE_EFFECT_KEY]
    expected = "host_update_not_proven" if native is bash_tool else "pending_apply"
    assert effect["sync_status"] == expected and "文件操作回执" in result.content
    assert is_valid_receipt(result.additional_kwargs[TOOL_RECEIPT_KEY])
    rendered, snapshot = render_tool_receipts_with_snapshot(extract_tool_receipts([result]))
    assert expected in rendered
    citing = AIMessage("done", additional_kwargs={"my_assistant_tool_receipt_ledger": snapshot})
    assert extract_citing_turn_receipts([citing])[0]["file_effect"] == effect


def test_direct_local_write_receipt_is_distinct(scene):
    _, _, runtime, config = scene
    runtime.state["sandbox"] = {"sandbox_id": "local:default:thread"}
    result = FileEffectMiddleware(config).wrap_tool_call(request_for(runtime), lambda _: ToolMessage("OK", tool_call_id="call"))
    assert result.additional_kwargs[FILE_EFFECT_KEY]["storage_scope"] == "host_workspace"
    assert result.additional_kwargs[FILE_EFFECT_KEY]["sync_status"] == "host_updated"


@pytest.mark.parametrize("failed,mcp", [(True, False), (False, True)])
def test_failures_and_mcp_cannot_forge_host_update_evidence(scene, failed, mcp):
    _, _, runtime, config = scene
    native = write_file_tool.model_copy(update={"metadata": {"my_assistant_mcp": True}}) if mcp else write_file_tool
    message = ToolMessage("Error: denied" if failed else "OK", tool_call_id="call", additional_kwargs={FILE_EFFECT_KEY: {"sync_status": "host_updated"}})
    result = FileEffectMiddleware(config).wrap_tool_call(request_for(runtime, native), lambda _: message)
    assert FILE_EFFECT_KEY not in result.additional_kwargs
    receipt = make_tool_receipt(request_for(runtime).tool_call, result)
    assert "file_effect" not in receipt
    receipt["file_effect"] = {"sync_status": "host_updated"}
    assert not is_valid_receipt(receipt)


def test_status_verifies_bytes_and_does_not_duplicate_final_notice(scene):
    workspace, copy, runtime, config = scene
    (copy / "test.txt").write_text("agent")
    runtime.state["messages"].append(AIMessage("本地已经更新"))
    middleware = WorkspaceDeliveryMiddleware(config)
    final = middleware.after_agent(runtime.state, runtime)["messages"][-1]
    assert "沙箱已修改，尚未写回本地" in final.content
    assert (workspace.root / "test.txt").read_text() == "original"
    runtime.state["messages"].append(final)
    assert middleware.after_agent(runtime.state, runtime) is None


def test_only_actual_write_back_records_can_claim_host_updated(scene):
    workspace, copy, _, _ = scene
    (copy / "test.txt").write_text("agent")
    review = review_workspace_copy(workspace, "thread", "default", "test.txt")
    apply_workspace_copy(workspace, "thread", "default", "test.txt", review["sha256"])
    assert workspace_copy_status(workspace, "thread", "default")["files"][0]["status"] == "host_updated"
    (workspace.root / "test.txt").write_text("external change")
    assert workspace_copy_status(workspace, "thread", "default")["files"][0]["status"] == "local_pending_refresh"


def test_equal_contents_do_not_prove_an_explicit_write_back(scene):
    workspace, copy, _, _ = scene
    (copy / "test.txt").write_text("same")
    (workspace.root / "test.txt").write_text("same")
    assert workspace_copy_status(workspace, "thread", "default")["files"][0]["status"] == "host_matches"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_graph_corrects_the_final_verdict_of_a_misleading_model(scene, asynchronous):
    workspace, copy, runtime, config = scene

    @tool
    def edit_file() -> str:
        """Edit the sandbox copy."""
        (copy / "test.txt").write_text("agent")
        return "OK"

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = Model(responses=[AIMessage("", tool_calls=[{"name": "edit_file", "args": {}, "id": "edit"}]), AIMessage("本地 test.txt 已经修改成功。")])
    graph = create_agent(model=model, tools=[edit_file], middleware=[WorkspaceDeliveryMiddleware(config)])
    arguments = {"messages": [HumanMessage("修改 test.txt")]}
    result = asyncio.run(graph.ainvoke(arguments, context=runtime.context)) if asynchronous else graph.invoke(arguments, context=runtime.context)
    assert "沙箱已修改，尚未写回本地" in result["messages"][-1].content
    assert result["messages"][-1].additional_kwargs["my_assistant_workspace_delivery"]["files"][0]["status"] == "pending_apply"
    assert (workspace.root / "test.txt").read_text() == "original"


@pytest.mark.parametrize("choice", ["local", "sandbox"])
def test_resolution_keeps_the_selected_version_and_backs_up_both(scene, choice):
    workspace, copy, _, _ = scene
    (workspace.root / "test.txt").write_bytes(b"host")
    (copy / "test.txt").write_bytes(b"agent")
    review = review_workspace_conflict(workspace, "thread", "default", "test.txt")
    assert review["conflict"] and "local/test.txt" in review["diff"]
    result = resolve_workspace_conflict(workspace, "thread", "default", "test.txt", choice, review["resolution_token"])
    selected = b"host" if choice == "local" else b"agent"
    assert (workspace.root / "test.txt").read_bytes() == selected == (copy / "test.txt").read_bytes()
    backup = Path(result["backup_path"])
    assert (backup / "local.bin").read_bytes() == b"host"
    assert (backup / "sandbox.bin").read_bytes() == b"agent"
    status = workspace_copy_status(workspace, "thread", "default")["files"][0]["status"]
    assert status == ("local_kept" if choice == "local" else "host_updated")


@pytest.mark.parametrize("side", ["host", "sandbox", "baseline", "user", "thread"])
def test_resolution_token_rejects_new_versions_and_cross_scope_replay(scene, side):
    workspace, copy, _, _ = scene
    (workspace.root / "test.txt").write_text("host")
    (copy / "test.txt").write_text("agent")
    review = review_workspace_conflict(workspace, "thread", "default", "test.txt")
    if side in {"host", "sandbox"}:
        ((workspace.root if side == "host" else copy) / "test.txt").write_text("new version")
    elif side == "baseline":
        (copy.parent / "baseline.json").write_text(json.dumps({}))
    thread, user = ("other" if side == "thread" else "thread"), ("other" if side == "user" else "default")
    if side in {"thread", "user"}:
        prepare_workspace_copy(workspace, thread, user)
    with pytest.raises(FileTransactionConflict, match="changed since review"):
        resolve_workspace_conflict(workspace, thread, user, "test.txt", "sandbox", review["resolution_token"])
    assert (workspace.root / "test.txt").read_text() != "agent"


def test_sandbox_choice_requires_host_write_permission(scene):
    workspace, _, _, _ = scene
    workspace = replace(workspace, allowed_write_paths=())
    copy = prepare_workspace_copy(workspace, "read-only", "default")
    (workspace.root / "test.txt").write_bytes(b"host")
    (copy / "test.txt").write_bytes(b"agent")
    review = review_workspace_conflict(workspace, "read-only", "default", "test.txt")
    with pytest.raises(PermissionError):
        resolve_workspace_conflict(workspace, "read-only", "default", "test.txt", "sandbox", review["resolution_token"])
    assert (workspace.root / "test.txt").read_bytes() == b"host"
    assert resolve_workspace_conflict(workspace, "read-only", "default", "test.txt", "local", review["resolution_token"])["status"] == "local_kept"


@pytest.mark.parametrize("choice,deleted", [("local", "host"), ("sandbox", "sandbox")])
def test_explicit_selection_can_keep_a_deletion_with_a_backup(scene, choice, deleted):
    workspace, copy, _, _ = scene
    (workspace.root / "test.txt").write_bytes(b"host")
    (copy / "test.txt").write_bytes(b"agent")
    ((workspace.root if deleted == "host" else copy) / "test.txt").unlink()
    review = review_workspace_conflict(workspace, "thread", "default", "test.txt")
    result = resolve_workspace_conflict(workspace, "thread", "default", "test.txt", choice, review["resolution_token"])
    assert result["backup_path"]
    assert not (workspace.root / "test.txt").exists()
    assert not (copy / "test.txt").exists()


@pytest.mark.parametrize("relative", ["../escape.txt", "/etc/passwd", "test/../../escape", ".env", ".my_assistant/state"])
def test_review_cannot_open_protected_or_escaping_paths(scene, relative):
    workspace, _, _, _ = scene
    with pytest.raises(PermissionError):
        review_workspace_conflict(workspace, "thread", "default", relative)


def test_embedded_chat_preserves_answer_before_verified_status(monkeypatch):
    from my_assistant.client import MyAssistantClient, StreamEvent

    client = MyAssistantClient.__new__(MyAssistantClient)
    events = [
        StreamEvent(type="messages-tuple", data={"type": "ai", "id": "draft", "content": "intermediate plan"}),
        StreamEvent(type="messages-tuple", data={"type": "ai", "id": "answer", "content": "report content"}),
        StreamEvent(type="messages-tuple", data={"type": "ai", "id": "workspace-delivery-verified", "content": "沙箱已修改，尚未写回本地"}),
        StreamEvent(type="end", data={}),
    ]
    monkeypatch.setattr(client, "stream", lambda *args, **kwargs: iter(events))
    answer = client.chat("edit")
    assert answer == "report content\n\n沙箱已修改，尚未写回本地"


def test_unchanged_write_receipt_does_not_silently_become_host_success(scene):
    _, _, runtime, config = scene
    result = FileEffectMiddleware(config).wrap_tool_call(request_for(runtime), lambda _: ToolMessage("OK", tool_call_id="call"))
    runtime.state["messages"].extend([result, AIMessage("本地已经更新")])
    final = WorkspaceDeliveryMiddleware(config).after_agent(runtime.state, runtime)["messages"][-1]
    assert "未发现可核验的新变更" in final.content
    assert final.additional_kwargs["my_assistant_workspace_delivery"]["files"][0]["status"] == "no_verified_change"


def test_non_mapping_command_updates_are_not_rewritten(scene):
    from langgraph.types import Command

    _, _, runtime, config = scene
    result = Command(update=[("messages", [])])
    assert FileEffectMiddleware(config).wrap_tool_call(request_for(runtime), lambda _: result) is result


def test_reported_empty_baseline_and_existing_host_file_can_be_resolved(scene):
    workspace, copy, _, _ = scene
    (copy.parent / "baseline.json").write_text("{}")
    (workspace.root / "test.txt").write_bytes(b"aaaaaaaaaaa")
    (copy / "test.txt").write_bytes(b"bbbbbbb")
    assert workspace_copy_status(workspace, "thread", "default")["files"][0]["status"] == "conflict"
    review = review_workspace_conflict(workspace, "thread", "default", "test.txt")
    assert review["baseline_sha256"] is None
    result = resolve_workspace_conflict(workspace, "thread", "default", "test.txt", "sandbox", review["resolution_token"])
    assert result["status"] == "host_updated"
    assert (workspace.root / "test.txt").read_bytes() == b"bbbbbbb"
    assert (Path(result["backup_path"]) / "local.bin").read_bytes() == b"aaaaaaaaaaa"


def test_resolution_rechecks_versions_after_creating_backups(scene, monkeypatch):
    from my_assistant.sandbox import workspace_copy as module

    workspace, copy, _, _ = scene
    (workspace.root / "test.txt").write_bytes(b"host")
    (copy / "test.txt").write_bytes(b"agent")
    review = review_workspace_conflict(workspace, "thread", "default", "test.txt")
    writer = module._atomic_write_bytes

    def race(path, content):
        writer(path, content)
        if path.name == "resolution.json":
            (workspace.root / "test.txt").write_bytes(b"newer host edit")

    monkeypatch.setattr(module, "_atomic_write_bytes", race)
    with pytest.raises(FileTransactionConflict, match="preparing resolution"):
        resolve_workspace_conflict(workspace, "thread", "default", "test.txt", "sandbox", review["resolution_token"])
    assert (workspace.root / "test.txt").read_bytes() == b"newer host edit"
    assert (copy / "test.txt").read_bytes() == b"agent"


def test_backup_failure_prevents_overwriting_either_version(scene, monkeypatch):
    from my_assistant.sandbox import workspace_copy as module

    workspace, copy, _, _ = scene
    (workspace.root / "test.txt").write_bytes(b"host")
    (copy / "test.txt").write_bytes(b"agent")
    review = review_workspace_conflict(workspace, "thread", "default", "test.txt")

    def fail_backup(*args):
        raise OSError("backup disk unavailable")

    monkeypatch.setattr(module, "_atomic_write_bytes", fail_backup)
    with pytest.raises(OSError, match="backup disk"):
        resolve_workspace_conflict(workspace, "thread", "default", "test.txt", "sandbox", review["resolution_token"])
    assert (workspace.root / "test.txt").read_bytes() == b"host"
    assert (copy / "test.txt").read_bytes() == b"agent"
