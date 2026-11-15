from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallRequest
from langchain.agents.middleware.todo import write_todos
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from my_assistant.agents.middlewares.file_transaction_recovery_middleware import (
    FileTransactionRecoveryMiddleware,
)
from my_assistant.agents.middlewares.task_workflow_middleware import (
    TaskWorkflowMiddleware,
    plan_submit,
)
from my_assistant.agents.middlewares.todo_middleware import TodoMiddleware
from my_assistant.agents.middlewares.token_budget_middleware import (
    TokenBudgetMiddleware,
)
from my_assistant.agents.task_continuity.task_state import (
    create_task_state,
    update_task_progress,
)
from my_assistant.agents.task_continuity.tools import task_update
from my_assistant.agents.task_continuity.workflow import (
    approve,
    begin_preview,
    check_delivery,
    submit,
    tool_allowed,
)
from my_assistant.agents.thread_state import get_thread_state_schema
from my_assistant.client import MyAssistantClient
from my_assistant.config.token_budget_config import TokenBudgetConfig
from my_assistant.config.workspace import Workspace
from my_assistant.sandbox.tools import read_file_tool
from my_assistant.tui.app import MyAssistantTUI
from my_assistant.tui.cli import LaunchPlan
from my_assistant.tui.runtime import restore_view_state
from my_assistant.tui.view_state import ViewState
from my_assistant.workspaces import WorkspaceRegistry, workspace_binding


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def _plan_args():
    return {"steps": ["inspect", "deliver"], "allowed_actions": [{"tool": "write_file", "args": {"path": "report.md"}}],
            "acceptance_criteria": [{"kind": "human", "description": "Compare costs and risks with sources"}]}


def _saved_task():
    return submit(begin_preview(create_task_state("deliver report")), ["inspect", "deliver"], _plan_args()["allowed_actions"],
                  [{"id": "c1", "kind": "human", "description": "Compare costs and risks with sources"}])


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_graph_stops_after_plan_submission_without_sibling_effect(tmp_path, asynchronous):
    effects = []

    def write(path: str, content: str) -> str:
        """Write a report (test double)."""
        effects.append((path, content))
        return "OK"

    message = _call("plan_submit", _plan_args(), "plan-1")
    message.tool_calls.append({"name": "write_file", "args": {"path": "report.md", "content": "bad"}, "id": "write-1", "type": "tool_call"})
    model = ScriptedModel(responses=[message, AIMessage(content="must not reach this response")])
    graph = create_agent(model, tools=[plan_submit, task_update, StructuredTool.from_function(write, name="write_file")],
                         middleware=[TaskWorkflowMiddleware(), TodoMiddleware()], state_schema=get_thread_state_schema("full"), checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "preview"}, "recursion_limit": 25}
    context = {"thread_id": "preview", "plan_preview": True, "workspace": Workspace.from_path(tmp_path).to_context()}
    payload = {"messages": [HumanMessage("prepare a plan")]}
    result = asyncio.run(graph.ainvoke(payload, config=config, context=context)) if asynchronous else graph.invoke(payload, config=config, context=context)
    assert effects == [] and model.i == 1
    assert result["task"]["execution"]["phase"] == "awaiting_approval"
    assert [row["content"] for row in result["todos"]] == ["inspect", "deliver"]
    assert graph.get_state(config).next == ()
    assert "等待确认" in result["messages"][-1].content


def test_full_preview_approve_continue_deliver_accept_cycle_survives_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_ASSISTANT_HOME", str(tmp_path / "state"))
    effects = []

    def write(path: str, content: str) -> str:
        """Create the approved report (test double)."""
        (tmp_path / path).write_text(content, encoding="utf-8")
        effects.append(path)
        return "OK"

    args = _plan_args()
    args["acceptance_criteria"].append({"kind": "file_nonempty", "path": "report.md", "description": "non-empty report"})
    model = ScriptedModel(responses=[
        _call("plan_submit", args, "plan"),
        _call("write_file", {"path": "report.md", "content": "Costs, risks and cited sources."}, "write"),
        _call("write_todos", {"todos": [{"content": "inspect", "status": "completed"}, {"content": "deliver", "status": "completed"}]}, "todo"),
        _call("task_update", {"status": "completed"}, "deliver"),
        AIMessage(content="Report delivered for your review."),
    ])
    database = tmp_path / "workflow.sqlite"
    workspace = Workspace.from_path(tmp_path)
    with SqliteSaver.from_conn_string(str(database)) as saver:
        graph = create_agent(model, tools=[plan_submit, task_update, StructuredTool.from_function(write, name="write_file")],
                             middleware=[TaskWorkflowMiddleware(), TodoMiddleware()], state_schema=get_thread_state_schema("full"), checkpointer=saver)
        config = {"configurable": {"thread_id": "cycle"}, "recursion_limit": 60}
        context = {"thread_id": "cycle", "plan_preview": True, "workspace": workspace.to_context()}
        result = graph.invoke({"messages": [HumanMessage("plan report")], "workspace_binding": workspace_binding(workspace)}, config=config, context=context)
        assert effects == [] and result["task"]["execution"]["phase"] == "awaiting_approval"
    with SqliteSaver.from_conn_string(str(database)) as saver:
        # A fresh process starts with no in-memory graph. Historical branch
        # versions must not be mistaken for an interrupted execution.
        client = MyAssistantClient.__new__(MyAssistantClient)
        client._checkpointer = saver
        client._checkpoint_channel_mode = "full"
        client._checkpoint_snapshot_frequency = 3
        client._agent = None
        client.approve_plan("cycle", 1, workspace=workspace.to_context())
        client.resume_execution("cycle", workspace=workspace.to_context())
        graph.checkpointer = saver
        result = graph.invoke({"messages": [HumanMessage("continue", additional_kwargs={"hide_from_ui": True})]}, config=config,
                              context={"thread_id": "cycle", "resume_task": True, "workspace": workspace.to_context()})
        assert effects == ["report.md"]
        assert result["task"]["status"] == "delivered"
        assert [c["status"] for c in result["task"]["execution"]["checks"]] == ["unverified", "passed"]
        accepted = client.check_task_acceptance("cycle", workspace=workspace.to_context(), confirm=True)
        assert accepted["task"]["status"] == "completed"


def test_simple_chat_creates_no_task_or_todos_even_with_plan_switch():
    graph = create_agent(ScriptedModel(responses=[AIMessage(content="Hello")]), tools=[plan_submit, task_update],
                         middleware=[TaskWorkflowMiddleware(), TodoMiddleware()], state_schema=get_thread_state_schema("full"))
    result = graph.invoke({"messages": [HumanMessage("hello")]}, context={"thread_id": "chat"})
    assert result.get("task") is None and not result.get("todos")


def test_real_token_budget_guard_records_blocker():
    task, todos = _saved_task()
    task = approve(task, todos, 1)
    model = ScriptedModel(responses=[AIMessage(content="partial result", usage_metadata={"input_tokens": 1000, "output_tokens": 10, "total_tokens": 1010})])
    graph = create_agent(model, tools=[plan_submit, task_update], middleware=[TaskWorkflowMiddleware(),
                         TokenBudgetMiddleware(TokenBudgetConfig(enabled=True, max_tokens=1000, hard_stop_threshold=1.0))],
                         state_schema=get_thread_state_schema("full"))
    result = graph.invoke({"messages": [HumanMessage("run")], "task": task, "todos": todos}, context={"thread_id": "budget", "run_id": "budget-run"})
    assert result["task"]["status"] == "blocked"
    assert result["task"]["execution"]["blocker"]["reason"] == "token_capped"


@pytest.mark.parametrize("mode", ["full", "delta"])
def test_plan_approval_and_todos_survive_sqlite_restart(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("MY_ASSISTANT_HOME", str(tmp_path / "state"))
    root = tmp_path / "project"
    root.mkdir()
    registry = WorkspaceRegistry(tmp_path / "state" / "users" / "default" / "workspaces")
    workspace = registry.register(root)
    database = tmp_path / "checkpoint.sqlite"
    task, todos = _saved_task()

    def client_for(saver):
        client = MyAssistantClient.__new__(MyAssistantClient)
        client._checkpointer = saver
        client._checkpoint_channel_mode = mode
        client._checkpoint_snapshot_frequency = 3
        client._agent = None
        return client

    with SqliteSaver.from_conn_string(str(database)) as saver:
        client = client_for(saver)
        binding = client.check_thread_workspace("saved-plan", workspace=workspace.to_context(), bind=True)
        task["workspace_binding"] = binding
        accessor, config, _ = client._workspace_state("saved-plan")
        accessor.update(config, {"workspace_binding": binding, "task": task, "todos": todos}, as_node="workspace_control")
    with SqliteSaver.from_conn_string(str(database)) as saver:
        client = client_for(saver)
        assert client.get_execution("saved-plan")["todos"] == todos
        approved = client.approve_plan("saved-plan", 1, workspace=workspace.to_context())
        assert approved["task"]["execution"]["approval"]["source"] == "user"
        with pytest.raises(ValueError, match="accept"):
            client.update_task("saved-plan", {}, status="completed")
    with SqliteSaver.from_conn_string(str(database)) as saver:
        saved = client_for(saver).get_execution("saved-plan")
        assert tool_allowed(saved["task"], saved["todos"], "write_file", {"path": "report.md", "content": "x"})
        restored = restore_view_state({"checkpoints": [{"values": saved}]})
        assert restored.task["execution"]["phase"] == "executing" and len(restored.todos) == 2


def test_policy_rejects_mcp_impersonation_and_changed_shell_arguments():
    task, todos = _saved_task()
    middleware = TaskWorkflowMiddleware()
    fake_read = read_file_tool.model_copy(update={"metadata": {"my_assistant_mcp": True}})
    request = ToolCallRequest(tool_call={"name": "read_file", "args": {"path": "x"}, "id": "read"}, tool=fake_read,
                              state={"task": task, "todos": todos}, runtime=SimpleNamespace(context={}))
    calls = []
    assert middleware.wrap_tool_call(request, calls.append).status == "error" and calls == []
    task, todos = submit(task, ["inspect"], [{"tool": "bash", "args": {"command": "python -m pytest", "description": "test"}}], task["acceptance_criteria"])
    task = approve(task, todos, task["execution"]["plan_version"])
    assert tool_allowed(task, todos, "bash", {"command": "python -m pytest", "description": "test"})
    assert not tool_allowed(task, todos, "bash", {"command": "python -m pytest; echo bad", "description": "test"})


def test_todo_status_keeps_approval_but_content_change_revokes_it():
    task, todos = _saved_task()
    task = approve(task, todos, 1)
    middleware = TaskWorkflowMiddleware()
    changed = [{**row, "status": "completed"} for row in todos]
    request = ToolCallRequest(tool_call={"name": "write_todos", "args": {"todos": changed}, "id": "todo"}, tool=write_todos,
                              state={"task": task, "todos": todos}, runtime=SimpleNamespace(context={}))
    result = middleware.wrap_tool_call(request, lambda _: pytest.fail("must project atomically"))
    assert isinstance(result, Command)
    assert result.update["task"]["execution"]["plan_version"] == 1
    assert result.update["task"]["completed_steps"] == ["inspect", "deliver"]
    changed.append({"content": "new scope", "status": "pending"})
    result = middleware.wrap_tool_call(request, lambda _: None)
    assert result.update["task"]["execution"]["plan_version"] == 2
    assert result.update["task"]["execution"]["approval"] is None
    assert result.update["task"]["execution"]["phase"] == "awaiting_approval"


def test_persistent_errors_stop_before_another_model_call():
    task, todos = _saved_task()
    task = approve(task, todos, 1)
    errors = [ToolMessage(content="Error: missing", status="error", tool_call_id=f"e{i}") for i in range(3)]
    update = TaskWorkflowMiddleware().before_model({"task": task, "todos": todos, "messages": [HumanMessage("run"), *errors]}, SimpleNamespace(context={}))
    assert update["jump_to"] == "end" and update["task"]["execution"]["phase"] == "blocked"
    assert update["task"]["execution"]["blocker"]["options"]


def test_budget_or_unfinished_todos_cannot_be_claimed_as_completed(tmp_path):
    task, todos = _saved_task()
    task = update_task_progress(approve(task, todos, 1), {"status": "completed"})
    middleware = TaskWorkflowMiddleware()
    runtime = SimpleNamespace(context={"thread_id": "budget", "stop_reason": "token_capped", "workspace": Workspace.from_path(tmp_path).to_context()})
    update = middleware.after_agent({"task": task, "todos": todos, "messages": []}, runtime)
    assert update["task"]["execution"]["blocker"]["reason"] == "token_capped"
    runtime.context.pop("stop_reason")
    update = middleware.after_agent({"task": task, "todos": todos, "messages": []}, runtime)
    assert update["task"]["status"] == "blocked"
    assert check_delivery(task, None, "budget", human_confirm=True, todos=todos)["status"] == "delivered"


def test_preview_does_not_replay_file_recovery(monkeypatch):
    monkeypatch.setattr("my_assistant.agents.middlewares.file_transaction_recovery_middleware.FileTransactionStore", lambda *args, **kwargs: pytest.fail("preview must not replay recovery"))
    task = begin_preview(create_task_state("plan"))
    assert FileTransactionRecoveryMiddleware().before_agent({"task": task}, SimpleNamespace(context={"thread_id": "read-only", "resume_file_writes": True})) is None


@pytest.mark.asyncio
async def test_tui_plan_switch_approval_and_acceptance_commands():
    task, todos = _saved_task()

    class Client:
        def list_models(self):
            return {"models": []}

        def list_skills(self, **kwargs):
            return {"skills": []}

        def get_execution(self, thread_id):
            return {"task": task, "todos": todos}

        def approve_plan(self, thread_id, version, **kwargs):
            nonlocal task
            task = approve(task, todos, version)
            return {"task": task, "todos": todos}

        def check_task_acceptance(self, thread_id, **kwargs):
            nonlocal task
            task = check_delivery(task, None, thread_id, human_confirm=kwargs["confirm"], todos=todos)
            return {"task": task, "todos": todos}

    app = MyAssistantTUI(SimpleNamespace(client=Client(), resolve_thread=lambda _: "ui-plan", load_view_state=lambda _: ViewState()), LaunchPlan(mode="tui"))
    async with app.run_test(size=(120, 40)):
        app._handle_plan("on")
        assert app._plan_enabled
        app._handle_plan("preview")
        assert app._preview_next
        app._handle_plan("show")
        assert app.state.task["execution"]["phase"] == "awaiting_approval"
        app._handle_plan("approve 1")
        assert app.state.task["execution"]["phase"] == "executing"
        todos[:] = [{**row, "status": "completed"} for row in todos]
        task["status"], task["execution"]["phase"] = "delivered", "delivered"
        app._handle_accept("check")
        assert app.state.task["status"] == "delivered"
        app._handle_accept("confirm")
        assert app.state.task["status"] == "completed"
        app._handle_plan("off")
        assert not app._plan_enabled and not app._preview_next
