from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from my_assistant.agents.middlewares.task_workflow_middleware import (
    TaskWorkflowMiddleware,
    plan_submit,
)
from my_assistant.agents.middlewares.tool_receipt import (
    TOOL_RECEIPT_KEY,
    is_valid_receipt,
)
from my_assistant.agents.task_continuity.task_state import (
    create_task_state,
    update_task_progress,
)
from my_assistant.agents.task_continuity.workflow import (
    approve,
    begin_preview,
    check_delivery,
    prepare_criteria,
    submit,
    tool_allowed,
    wants_preview,
)
from my_assistant.config.workspace import Workspace
from my_assistant.sandbox.tools import bash_tool, delete_file_tool, write_file_tool
from my_assistant.tui.runtime import restore_view_state, translate
from my_assistant.tui.view_state import ProgressUpdated


def _runtime(workspace: Workspace, thread_id: str = "workflow-test"):
    return SimpleNamespace(context={"thread_id": thread_id, "workspace": workspace.to_context()})


def _contract(tmp_path):
    workspace = Workspace.from_path(tmp_path)
    task = begin_preview(create_task_state("produce a report"))
    criteria = prepare_criteria(
        [
            {"kind": "file_exists", "path": "report.md", "description": "report exists"},
            {"kind": "human", "description": "conclusions and sources are sufficient"},
        ],
        workspace,
        "workflow-test",
    )
    task, todos = submit(
        task,
        ["inspect source", "write report"],
        [{"tool": "write_file", "args": {"path": "report.md"}}],
        criteria,
    )
    return workspace, task, todos


def test_simple_question_does_not_enter_preview() -> None:
    assert wants_preview("what is a workspace?") is False
    assert wants_preview("先回答我这个概念，不用执行") is False
    assert wants_preview("先规划，等我确认后再执行") is True
    assert wants_preview("plan first and wait for my approval before executing") is True


def test_preview_blocks_effectful_tools_before_handler(tmp_path) -> None:
    workspace, task, todos = _contract(tmp_path)
    middleware = TaskWorkflowMiddleware()
    runtime = _runtime(workspace)

    for tool in (write_file_tool, delete_file_tool, bash_tool):
        request = SimpleNamespace(
            state={"task": task, "todos": todos},
            tool_call={"id": "call-1", "name": tool.name, "args": {"path": "report.md", "command": "echo bad"}},
            tool=tool,
            runtime=runtime,
        )
        called = []
        result = middleware.wrap_tool_call(request, called.append)
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert called == []

    assert plan_submit.tool_call_schema.model_fields.keys() == {"steps", "allowed_actions", "acceptance_criteria"}


@pytest.mark.parametrize("asynchronous", [False, True])
def test_policy_denial_records_tool_receipt(tmp_path, asynchronous) -> None:
    workspace, task, todos = _contract(tmp_path)
    request = SimpleNamespace(
        state={"task": task, "todos": todos},
        tool_call={"id": "call-denied", "name": "write_file", "args": {"path": "report.md", "content": "blocked"}},
        tool=write_file_tool,
        runtime=_runtime(workspace),
    )
    middleware = TaskWorkflowMiddleware()

    if asynchronous:
        async def handler(_):
            pytest.fail("denied tool must not reach its handler")

        result = asyncio.run(middleware.awrap_tool_call(request, handler))
    else:
        result = middleware.wrap_tool_call(request, lambda _: pytest.fail("denied tool must not reach its handler"))

    assert result.status == "error"
    assert is_valid_receipt(result.additional_kwargs[TOOL_RECEIPT_KEY])


def test_approval_is_bound_to_version_digest_and_exact_action(tmp_path) -> None:
    _, task, todos = _contract(tmp_path)
    assert task["execution"]["phase"] == "awaiting_approval"
    assert not tool_allowed(task, todos, "write_file", {"path": "report.md", "content": "x"})

    approved = approve(task, todos, task["execution"]["plan_version"])
    assert approved["execution"]["phase"] == "executing"
    assert tool_allowed(approved, todos, "write_file", {"path": "report.md", "content": "x"})
    assert not tool_allowed(approved, todos, "write_file", {"path": "other.md", "content": "x"})
    with pytest.raises(ValueError, match="stale|changed"):
        approve(task, todos, task["execution"]["plan_version"] - 1)


def test_plan_revision_revokes_previous_approval(tmp_path) -> None:
    _, task, todos = _contract(tmp_path)
    task = approve(task, todos, task["execution"]["plan_version"])
    revised, revised_todos = submit(
        task,
        ["inspect source", "write report", "verify report"],
        task["execution"]["allowed_actions"],
        task["acceptance_criteria"],
    )
    assert revised_todos[-1]["status"] == "pending"
    assert revised["execution"]["plan_version"] == task["execution"]["plan_version"] + 1
    assert revised["execution"]["approval"] is None
    assert not tool_allowed(revised, revised_todos, "write_file", {"path": "report.md", "content": "x"})


def test_delivery_requires_code_checks_and_human_confirmation(tmp_path) -> None:
    workspace, task, todos = _contract(tmp_path)
    (tmp_path / "report.md").write_text("report", encoding="utf-8")
    completed = [{**row, "status": "completed"} for row in todos]

    delivered = check_delivery(task, workspace, "workflow-test", todos=completed)
    assert delivered["status"] == "delivered"
    assert [item["status"] for item in delivered["execution"]["checks"]] == ["passed", "unverified"]

    accepted = check_delivery(delivered, workspace, "workflow-test", todos=completed, human_confirm=True)
    assert accepted["status"] == "completed"
    assert accepted["execution"]["phase"] == "completed"


def test_file_unchanged_acceptance_detects_external_change(tmp_path) -> None:
    workspace = Workspace.from_path(tmp_path)
    source = tmp_path / "source.md"
    source.write_text("before", encoding="utf-8")
    task = begin_preview(create_task_state("compare source"))
    criteria = prepare_criteria(
        [{"kind": "file_unchanged", "path": "source.md", "description": "source remains unchanged"}],
        workspace,
        "workflow-test",
    )
    task, _ = submit(task, ["compare"], [], criteria)
    source.write_text("after", encoding="utf-8")
    result = check_delivery(task, workspace, "workflow-test", todos=[{"content": "compare", "status": "completed"}], human_confirm=True)
    assert result["status"] == "delivered"
    assert result["execution"]["checks"][0]["status"] == "failed"


def test_model_completion_stays_delivered_until_acceptance() -> None:
    task = begin_preview(create_task_state("deliver safely"))
    task, _ = submit(task, ["deliver"], [], [{"id": "c1", "kind": "human", "description": "quality review"}])
    updated = update_task_progress(task, {"status": "completed"})
    assert updated["status"] == "delivered"
    assert updated["execution"]["phase"] == "delivered"


def test_stream_and_restore_project_todos_and_task() -> None:
    task = {**create_task_state("restore me"), "execution": {"phase": "executing", "plan_version": 2, "require_approval": False}}
    todos = [{"content": "first", "status": "completed"}, {"content": "second", "status": "in_progress"}]
    action = translate(SimpleNamespace(type="values", data={"todos": todos, "task": task}))
    assert isinstance(action[0], ProgressUpdated)
    assert [item.content for item in action[0].todos] == ["first", "second"]
    restored = restore_view_state({"checkpoints": [{"values": {"todos": todos, "task": task}}]})
    assert [item.status for item in restored.todos] == ["completed", "in_progress"]
    assert restored.task["objective"] == "restore me"


def test_blocked_task_is_read_only_until_user_continues(tmp_path) -> None:
    workspace, task, todos = _contract(tmp_path)
    task["execution"]["phase"] = "blocked"
    task["execution"]["blocker"] = {"reason": "missing source", "options": ["add source"]}
    middleware = TaskWorkflowMiddleware()
    request = SimpleNamespace(state={"task": task, "todos": todos}, runtime=_runtime(workspace))
    update = middleware.before_model(request.state, request.runtime)
    assert update["jump_to"] == "end"
    assert "missing source" in update["task"]["execution"]["blocker"]["reason"]
