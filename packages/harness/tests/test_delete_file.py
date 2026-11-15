from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime as GraphRuntime
from my_assistant.agents.human_input import has_action_confirmation
from my_assistant.agents.middlewares.clarification_middleware import (
    ClarificationMiddleware,
)
from my_assistant.sandbox.local.local_sandbox import LocalSandbox, PathMapping
from my_assistant.sandbox.tools import delete_file_tool


def _confirmation_messages(path: str, *, answer: str = "\u786e\u8ba4\u5220\u9664", option_id: str = "option-1"):
    request_id = "clarification:delete-request"
    return [
        ToolMessage(
            content="Confirm deletion",
            tool_call_id="delete-request",
            name="delete_file",
            artifact={
                "human_input": {
                    "version": 1,
                    "kind": "human_input_request",
                    "source": "ask_clarification",
                    "request_id": request_id,
                    "clarification_type": "risk_confirmation",
                    "action": "delete_file",
                    "target_path": path,
                    "approval_option_id": "option-1",
                }
            },
        ),
        HumanMessage(
            content=answer,
            additional_kwargs={
                "human_input_response": {
                    "version": 1,
                    "kind": "human_input_response",
                    "source": "ask_clarification",
                    "request_id": request_id,
                    "response_kind": "option",
                    "option_id": option_id,
                    "value": answer,
                }
            },
        ),
    ]


def test_confirmation_is_fresh_explicit_and_bound_to_exact_path():
    path = "/mnt/user-data/workspace/notes.txt"
    messages = _confirmation_messages(path)

    assert has_action_confirmation(messages, action="delete_file", target=path)
    assert not has_action_confirmation(messages, action="delete_file", target=path + ".bak")
    assert not has_action_confirmation(
        _confirmation_messages(path, answer="\u53d6\u6d88", option_id="option-2"),
        action="delete_file",
        target=path,
    )
    assert not has_action_confirmation(
        messages + [HumanMessage(content="Actually, keep it")],
        action="delete_file",
        target=path,
    )


def test_plain_text_confirmation_supports_tui_but_not_stale_or_unrelated_yes():
    path = "/mnt/user-data/workspace/notes.txt"
    request = _confirmation_messages(path)[0]

    assert has_action_confirmation(
        [request, HumanMessage(content="\u786e\u8ba4\u5220\u9664")],
        action="delete_file",
        target=path,
    )
    assert not has_action_confirmation(
        [HumanMessage(content="yes")],
        action="delete_file",
        target=path,
    )
    assert not has_action_confirmation(
        [request, HumanMessage(content="no"), HumanMessage(content="yes")],
        action="delete_file",
        target=path,
    )


def test_confirmation_is_consumed_after_successful_deletion():
    path = "/mnt/user-data/workspace/notes.txt"
    messages = _confirmation_messages(path)
    messages.append(
        ToolMessage(
            content=f"Deleted file: {path}",
            tool_call_id="delete-result",
            name="delete_file",
            artifact={
                "delete_file_confirmation": {
                    "request_id": "clarification:delete-request",
                    "target_path": path,
                }
            },
        )
    )

    assert not has_action_confirmation(messages, action="delete_file", target=path)
    assert not has_action_confirmation(
        messages + [HumanMessage(content="yes")],
        action="delete_file",
        target=path,
    )


def test_delete_tool_is_interrupted_with_target_bound_confirmation():
    middleware = ClarificationMiddleware()
    path = "/mnt/user-data/workspace/notes.txt"
    request = SimpleNamespace(
        tool_call={"name": "delete_file", "args": {"path": path, "description": "Remove the obsolete note"}, "id": "delete-call"},
        runtime=SimpleNamespace(context={}),
        state={"messages": []},
    )
    called = False

    def handler(_request):
        nonlocal called
        called = True
        return ToolMessage(content="Deleted", tool_call_id="delete-call", name="delete_file")

    result = middleware.wrap_tool_call(request, handler)

    assert not called
    assert result.goto == "__end__"
    prompt = result.update["messages"][0]
    payload = prompt.artifact["human_input"]
    assert payload["action"] == "delete_file"
    assert payload["target_path"] == path
    assert payload["clarification_type"] == "risk_confirmation"
    assert "Remove the obsolete note" in payload["context"]
    assert payload["options"][0]["value"] == "\u786e\u8ba4\u5220\u9664"


def test_confirmed_tool_call_runs_once_and_stamps_consumed_approval():
    middleware = ClarificationMiddleware()
    path = "/mnt/user-data/workspace/notes.txt"
    messages = _confirmation_messages(path)
    request = SimpleNamespace(
        tool_call={"name": "delete_file", "args": {"path": path}, "id": "delete-call"},
        runtime=SimpleNamespace(context={}),
        state={"messages": messages},
    )
    result = middleware.wrap_tool_call(
        request,
        lambda _: ToolMessage(content=f"Deleted file: {path}", tool_call_id="delete-call", name="delete_file"),
    )

    assert result.artifact["delete_file_confirmation"] == {
        "request_id": "clarification:delete-request",
        "target_path": path,
    }
    assert not has_action_confirmation(
        messages + [result],
        action="delete_file",
        target=path,
    )


def test_failed_delete_attempt_also_consumes_approval():
    middleware = ClarificationMiddleware()
    path = "/mnt/user-data/workspace/notes.txt"
    messages = _confirmation_messages(path)
    request = SimpleNamespace(
        tool_call={"name": "delete_file", "args": {"path": path}, "id": "delete-call"},
        runtime=SimpleNamespace(context={}),
        state={"messages": messages},
    )
    result = middleware.wrap_tool_call(
        request,
        lambda _: ToolMessage(content="Error: file not found", tool_call_id="delete-call", name="delete_file"),
    )

    assert result.artifact["delete_file_confirmation"] == {
        "request_id": "clarification:delete-request",
        "target_path": path,
    }
    assert not has_action_confirmation(
        messages + [result],
        action="delete_file",
        target=path,
    )


def test_noninteractive_delete_is_blocked_even_when_confirmation_is_disabled():
    middleware = ClarificationMiddleware()
    path = "/mnt/user-data/workspace/notes.txt"
    request = SimpleNamespace(
        tool_call={"name": "delete_file", "args": {"path": path}, "id": "delete-call"},
        runtime=SimpleNamespace(context={"disable_clarification": True}),
        state={"messages": []},
    )
    called = False

    def handler(_request):
        nonlocal called
        called = True
        return ToolMessage(content="Deleted", tool_call_id="delete-call", name="delete_file")

    result = middleware.wrap_tool_call(request, handler)

    assert not called
    assert isinstance(result, ToolMessage)
    assert "no file was deleted" in result.content


def test_after_model_drops_sibling_calls_until_delete_is_confirmed():
    middleware = ClarificationMiddleware()
    path = "/mnt/user-data/workspace/notes.txt"
    message = AIMessage(
        content="",
        tool_calls=[
            {"name": "delete_file", "args": {"path": path}, "id": "delete-call", "type": "tool_call"},
            {"name": "write_file", "args": {"path": path, "content": "x"}, "id": "write-call", "type": "tool_call"},
        ],
    )

    update = middleware.after_model({"messages": [HumanMessage(content="Delete this"), message]}, GraphRuntime(context={}))

    assert [call["name"] for call in update["messages"][0].tool_calls] == ["delete_file"]


def test_local_sandbox_deletes_only_regular_files_without_following_links(tmp_path: Path):
    sandbox = LocalSandbox("local-test", [PathMapping("/mnt/user-data", str(tmp_path))])
    target = tmp_path / "workspace" / "notes.txt"
    target.parent.mkdir()
    target.write_text("note", encoding="utf-8")
    sandbox.delete_file("/mnt/user-data/workspace/notes.txt")
    assert not target.exists()

    with pytest.raises(FileNotFoundError):
        sandbox.delete_file("/mnt/user-data/workspace/missing.txt")
    with pytest.raises(IsADirectoryError):
        sandbox.delete_file("/mnt/user-data/workspace")

    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = tmp_path / "workspace" / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("Symlink creation is unavailable on this platform")
    with pytest.raises(PermissionError):
        sandbox.delete_file("/mnt/user-data/workspace/link.txt")
    assert outside.read_text(encoding="utf-8") == "outside"


def test_delete_tool_requires_confirmation_before_touching_sandbox():
    path = "/mnt/user-data/workspace/notes.txt"
    runtime = SimpleNamespace(context={}, state={"messages": []})

    result = delete_file_tool.func(runtime, path)

    assert "confirmation is required" in result


def test_delete_tool_enforces_workspace_and_uses_sandbox_api(tmp_path: Path, monkeypatch):
    from my_assistant.sandbox import tools as sandbox_tools

    workspace = tmp_path / "workspace"
    uploads = tmp_path / "uploads"
    outputs = tmp_path / "outputs"
    workspace.mkdir()
    uploads.mkdir()
    outputs.mkdir()
    target = workspace / "notes.txt"
    target.write_text("note", encoding="utf-8")
    path = "/mnt/user-data/workspace/notes.txt"
    sandbox = LocalSandbox("local-test", [PathMapping("/mnt/user-data", str(tmp_path))])
    runtime = SimpleNamespace(
        context={"thread_id": "thread-delete"},
        state={
            "messages": _confirmation_messages(path),
            "sandbox": {"sandbox_id": "local"},
            "thread_data": {
                "workspace_path": str(workspace),
                "uploads_path": str(uploads),
                "outputs_path": str(outputs),
            },
        },
    )
    monkeypatch.setattr(sandbox_tools, "ensure_sandbox_initialized", lambda _runtime: sandbox)
    monkeypatch.setattr(sandbox_tools, "ensure_thread_directories_exist", lambda _runtime: None)

    result = delete_file_tool.func(runtime, path)

    assert result == f"Deleted file: {path}"
    assert not target.exists()


def test_async_confirmation_gate_does_not_call_delete_handler():
    middleware = ClarificationMiddleware()
    path = "/mnt/user-data/workspace/notes.txt"
    request = SimpleNamespace(
        tool_call={"name": "delete_file", "args": {"path": path}, "id": "delete-call"},
        runtime=SimpleNamespace(context={}),
        state={"messages": []},
    )
    called = False

    async def handler(_request):
        nonlocal called
        called = True
        return ToolMessage(content="Deleted", tool_call_id="delete-call", name="delete_file")

    result = asyncio.run(middleware.awrap_tool_call(request, handler))

    assert not called
    assert result.goto == "__end__"
