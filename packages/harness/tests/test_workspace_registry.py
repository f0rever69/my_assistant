from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from my_assistant.agents.middlewares.thread_data_middleware import ThreadDataMiddleware
from my_assistant.agents.task_continuity.task_state import create_task_state
from my_assistant.config.workspace import Workspace
from my_assistant.workspace_client import WorkspaceClientMixin
from my_assistant.workspace_transfers import (
    MAX_EXPORT_BYTES,
    export_file,
    reference_export,
)
from my_assistant.workspaces import (
    WorkspaceMismatch,
    WorkspaceRegistry,
    WorkspaceUnavailable,
    state_binding,
    workspace_binding,
)


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_ASSISTANT_HOME", str(tmp_path / "state"))
    registry = WorkspaceRegistry(tmp_path / "state" / "users" / "default" / "workspaces")
    roots = [tmp_path / name for name in ("alpha", "beta")]
    for root in roots:
        root.mkdir()
    return registry, registry.register(roots[0], name="Alpha"), registry.register(roots[1], name="Beta")


class LocalClient(WorkspaceClientMixin):
    def __init__(self, saver=None, mode="full"):
        self.saver = saver or InMemorySaver()
        self._checkpoint_channel_mode = mode
        self._checkpoint_snapshot_frequency = 3

    def _get_thread_checkpointer(self):
        return self.saver


def test_registration_is_canonical_stable_and_persistent(catalog):
    registry, alpha, _ = catalog
    again = registry.register(alpha.root / ".", name="ignored rename")
    restarted = WorkspaceRegistry(registry.root)
    assert again == alpha == restarted.resolve("Alpha")
    assert alpha.id.startswith("ws-") and len(alpha.id) == 35
    assert alpha.root_fingerprint
    (alpha.root / "new.txt").write_text("normal edit", encoding="utf-8")
    assert restarted.resolve(alpha.id) == alpha  # mtime is not directory identity


def test_policy_is_persistent_revokes_stale_context_and_cannot_widen_root(catalog):
    registry, alpha, _ = catalog
    updated = registry.update(alpha.id, name="Renamed", allowed_read_paths=["docs"], allowed_write_paths=[], ignored_patterns=["private/**"], shell_enabled=True)
    assert registry.resolve("Renamed") == updated
    assert updated.shell_enabled and not updated.network_enabled
    with pytest.raises(WorkspaceUnavailable, match="policy changed"):
        registry.validate(alpha)
    with pytest.raises(PermissionError):
        updated.check("main.py")
    with pytest.raises(PermissionError):
        updated.check("docs/test.md", write=True)
    with pytest.raises(PermissionError):
        updated.check("docs/.env")


@pytest.mark.parametrize("patch", [
    {"root_path": "elsewhere"}, {"id": "replacement"},
    {"allowed_read_paths": ["../outside"]}, {"allowed_write_paths": ["C:/secret"]},
    {"allowed_read_paths": "all"}, {"allowed_write_paths": [""]},
    {"shell_enabled": "true"}, {"network_enabled": 1}, {"name": ""}, {"name": 123},
])
def test_invalid_policies_fail_closed(catalog, patch):
    registry, alpha, _ = catalog
    with pytest.raises((ValueError, TypeError)):
        registry.update(alpha.id, **patch)
    assert registry.resolve(alpha.id) == alpha


def test_duplicate_names_require_full_id(catalog):
    registry, alpha, beta = catalog
    registry.update(beta.id, name=alpha.name)
    with pytest.raises(LookupError, match="ambiguous"):
        registry.resolve(alpha.name)
    assert registry.resolve(alpha.id) == alpha


@pytest.mark.parametrize("invalid", ["delete", "missing", "replace"])
def test_unavailable_roots_keep_metadata_and_never_rebind_by_path(catalog, invalid):
    registry, alpha, _ = catalog
    registry.check("saved-thread", {}, alpha, bind=True)
    if invalid == "delete":
        registry.delete(alpha.id)
    else:
        alpha.root.rename(alpha.root.with_name("old-alpha"))
        if invalid == "replace":
            alpha.root.mkdir()
    assert registry.thread_binding("saved-thread") == workspace_binding(alpha)
    assert not next(row for row in registry.list() if row["id"] == alpha.id)["available"]
    with pytest.raises(WorkspaceUnavailable):
        registry.resolve(alpha.id)
    if invalid != "missing":
        if invalid == "replace":
            with pytest.raises(WorkspaceUnavailable, match="replaced"):
                registry.register(alpha.root)
            registry.delete(alpha.id)
        fresh = registry.register(alpha.root, name=alpha.name)
        assert fresh.id != alpha.id
        with pytest.raises(WorkspaceMismatch):
            registry.check("saved-thread", {}, fresh)


def test_thread_binding_checks_sidecar_checkpoint_task_and_local_scope(catalog):
    registry, alpha, beta = catalog
    binding = registry.check("thread-a", {}, alpha, bind=True)
    assert registry.check("thread-a", {"workspace_binding": binding}, alpha) == binding
    for other in (beta, None):
        with pytest.raises(WorkspaceMismatch):
            registry.check("thread-a", {}, other)
    with pytest.raises(WorkspaceMismatch, match="disagree"):
        registry.check("thread-a", {"workspace_binding": workspace_binding(beta)}, alpha)
    with pytest.raises(WorkspaceMismatch, match="disagree"):
        state_binding({"workspace_binding": binding, "task": {"workspace_binding": workspace_binding(beta)}})
    registry.check("thread-local", {}, None, bind=True)
    with pytest.raises(WorkspaceMismatch):
        registry.check("thread-local", {}, alpha)


@pytest.mark.parametrize("binding", [[], "alpha", {}, {"version": 2, "workspace_id": None}, {"version": 1, "workspace_id": "alpha"}])
def test_malformed_bindings_are_rejected(binding):
    with pytest.raises(WorkspaceMismatch):
        state_binding({"workspace_binding": binding})


def test_legacy_workspace_without_fingerprint_requires_explicit_migration():
    with pytest.raises(WorkspaceMismatch, match="Legacy"):
        state_binding({"thread_data": {"workspace_policy": {"id": "old", "root_path": "old"}}})


def test_concurrent_first_binding_has_exactly_one_owner(catalog):
    registry, alpha, beta = catalog

    def bind(workspace):
        try:
            return registry.check("concurrent", {}, workspace, bind=True)
        except WorkspaceMismatch:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(bind, [alpha, beta]))
    assert sum(item is not None for item in outcomes) == 1
    assert registry.thread_binding("concurrent") in outcomes


@pytest.mark.parametrize("mode", ["full", "delta"])
def test_client_binding_checkpoint_reopen_and_task_migration(catalog, tmp_path, mode):
    registry, alpha, beta = catalog
    db_path = tmp_path / f"{mode}.sqlite3"
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        client = LocalClient(saver, mode)
        binding = client.check_thread_workspace("source", workspace=alpha.to_context(), bind=True)
        accessor, config, _ = client._workspace_state("source")
        task = {**create_task_state("finish alpha"), "workspace_binding": binding, "completed_steps": ["old done"], "pending_steps": ["test"], "constraints": ["no network"], "modified_files": ["old.txt"]}
        accessor.update(config, {"workspace_binding": binding, "task": task, "messages": [HumanMessage("alpha private", id="a")]}, as_node="workspace_control")
        # Admission is not a state mutation and must leave checkpoint scheduling untouched.
        old_head = saver.get_tuple(config).config
        client.check_thread_workspace("source", workspace=alpha.to_context(), bind=True)
        assert saver.get_tuple(config).config == old_head
        with pytest.raises(WorkspaceMismatch):
            client.check_thread_workspace("source", workspace=beta.to_context())
    with SqliteSaver.from_conn_string(str(db_path)) as saver:
        client = LocalClient(saver, mode)
        assert client.get_thread_workspace("source")["workspace_binding"] == binding
        registry.delete(alpha.id)  # metadata and explicit task fork remain available
        result = client.migrate_task_workspace("source", beta.to_context())
        _, _, target = client._workspace_state(result["thread_id"])
        _, _, source = client._workspace_state("source")
        assert [msg.content for msg in source["messages"]] == ["alpha private"]
        assert target.get("messages", []) == []
        assert target["task"]["status"] == "paused"
        assert target["task"]["pending_steps"] == ["test"]
        assert target["task"]["constraints"] == ["no network"]
        assert target["task"]["completed_steps"] == target["task"]["modified_files"] == []
        assert target["task"]["workspace_binding"] == workspace_binding(beta)
        assert target["workspace_binding"] == workspace_binding(beta)


def test_legacy_messages_cannot_silently_adopt_workspace(catalog):
    _, alpha, _ = catalog
    client = LocalClient()
    accessor, config, _ = client._workspace_state("legacy")
    accessor.update(config, {"messages": [HumanMessage("old", id="old")]}, as_node="workspace_control")
    with pytest.raises(WorkspaceMismatch):
        client.check_thread_workspace("legacy", workspace=alpha.to_context(), bind=True)


@pytest.mark.parametrize("async_tool", [False, True])
def test_middleware_persists_binding_and_rechecks_revocation_before_tool(catalog, async_tool):
    registry, alpha, beta = catalog
    middleware = ThreadDataMiddleware()
    runtime = SimpleNamespace(context={"thread_id": "middleware", "workspace": alpha.to_context()}, state={})
    output = middleware.before_agent({"messages": [HumanMessage("hello")]}, runtime)
    assert output["workspace_binding"] == workspace_binding(alpha)
    runtime.state = output
    request = SimpleNamespace(runtime=runtime, tool_call={"id": "call"})
    called = []
    registry.delete(alpha.id)
    if async_tool:
        async def handler(request):
            called.append(request)
        message = asyncio.run(middleware.awrap_tool_call(request, handler))
    else:
        message = middleware.wrap_tool_call(request, called.append)
    assert message.status == "error" and not called
    runtime.context["workspace"] = beta.to_context()
    with pytest.raises(WorkspaceMismatch):
        middleware.before_agent(output, runtime)
    assert not (beta.root / ".my-assistant").exists()


@pytest.mark.parametrize("bind_workspace", [False, True])
def test_present_files_uses_current_thread_outputs(catalog, bind_workspace):
    from my_assistant.tools.builtins.present_file_tool import present_file_tool

    _, alpha, _ = catalog
    context = {"thread_id": "present-file"}
    if bind_workspace:
        context["workspace"] = alpha.to_context()
    runtime = SimpleNamespace(context=context)
    runtime.state = ThreadDataMiddleware().before_agent({}, runtime)
    outputs = Path(runtime.state["thread_data"]["outputs_path"])
    artifact = outputs / "reports" / "bubble_sort.py"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("# generated file", encoding="utf-8")
    virtual = "/mnt/user-data/outputs/reports/bubble_sort.py"

    for path in (virtual, virtual.lstrip("/"), str(artifact)):
        result = present_file_tool.func(runtime, [path], "present")
        assert result.update.get("artifacts") == [virtual], result.update["messages"][0].content

    for path in (
        "/mnt/user-data/uploads/input.py",
        "/mnt/user-data/outputs/../uploads/input.py",
        "/mnt/user-data/outputs/",
        str(outputs.with_name("uploads") / "input.py"),
    ):
        result = present_file_tool.func(runtime, [path], "present")
        assert "artifacts" not in result.update
        assert "Only files in /mnt/user-data/outputs" in result.update["messages"][0].content


def test_explicit_export_reference_is_snapshot_scoped_and_idempotent(catalog):
    registry, alpha, beta = catalog
    path = alpha.root / "notes.md"
    path.write_text("alpha evidence", encoding="utf-8")
    client = LocalClient()
    manifest = client.export_workspace_file(alpha.to_context(), "notes.md")
    path.write_text("changed later", encoding="utf-8")
    registry.delete(alpha.id)
    result = client.reference_workspace_export("beta-thread", beta.to_context(), manifest["id"])
    target = beta.runtime_dir("beta-thread") / "uploads" / f"{manifest['id']}.txt"
    assert target.read_text(encoding="utf-8") == "alpha evidence"
    assert result["target_workspace_id"] == beta.id
    assert result["virtual_path"].startswith("/mnt/user-data/uploads/")
    assert client.reference_workspace_export("beta-thread", beta.to_context(), manifest["id"]) == result
    _, _, state = client._workspace_state("beta-thread")
    assert len(state["messages"]) == 1
    assert state["messages"][0].additional_kwargs["my_assistant_producer_kind"] == "workspace_reference"
    assert not (beta.root / "notes.md").exists()
    assert not (beta.root / ".my-assistant/index").exists()
    target.write_text("modified by user", encoding="utf-8")
    with pytest.raises(PermissionError, match="modified"):
        reference_export(registry, beta.to_context(), "beta-thread", manifest["id"])


@pytest.mark.parametrize(
    "name,content",
    [
        pytest.param(".env", "TOKEN=secret", id="env-file"),
        pytest.param("key.pem", "key", id="pem-file"),
        pytest.param("notes.md", "password=very-secret-value", id="secret-content"),
        pytest.param("binary.txt", "a\0b", id="binary-content"),
        pytest.param("large.txt", "x" * (MAX_EXPORT_BYTES + 1), id="size-limit"),
    ],
)
def test_unsafe_exports_are_rejected(catalog, name, content):
    registry, alpha, _ = catalog
    (alpha.root / name).write_text(content, encoding="utf-8")
    with pytest.raises((PermissionError, ValueError)):
        export_file(registry, alpha.to_context(), name)


def test_reference_integrity_user_isolation_and_publication_failure(catalog, tmp_path, monkeypatch):
    registry, alpha, beta = catalog
    (alpha.root / "safe.txt").write_text("safe", encoding="utf-8")
    manifest = export_file(registry, alpha.to_context(), "safe.txt")
    other = WorkspaceRegistry(tmp_path / "other-user")
    with pytest.raises(LookupError):
        reference_export(other, Workspace.from_path(beta.root).to_context(), "other", manifest["id"])
    monkeypatch.setattr("my_assistant.workspace_transfers.os.link", lambda *a, **kw: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError, match="interrupted"):
        reference_export(registry, beta.to_context(), "target", manifest["id"])
    assert not list((beta.runtime_dir("target") / "uploads").iterdir())
    with closing(registry._connect()) as db, db:
        db.execute("UPDATE workspace_exports SET content=? WHERE id=?", (b"tampered", manifest["id"]))
    with pytest.raises(PermissionError, match="integrity"):
        reference_export(registry, beta.to_context(), "target", manifest["id"])


@pytest.fixture
def ui_client(catalog, monkeypatch):
    from my_assistant.client import MyAssistantClient

    # Real state/registry APIs, no model, MCP, provider or network construction.
    client = MyAssistantClient.__new__(MyAssistantClient)
    client._checkpointer = InMemorySaver()
    client._checkpoint_channel_mode = "full"
    client._checkpoint_snapshot_frequency = 3
    client._agent = None
    monkeypatch.setattr(client, "list_models", lambda: {"models": []})
    monkeypatch.setattr(client, "list_skills", lambda **kw: {"skills": []})
    monkeypatch.setattr(client, "reset_agent", lambda: None)
    return client


def _ui_session(client):
    from my_assistant.tui.runtime import restore_view_state

    def load(thread_id):
        _, _, state = client._workspace_state(thread_id)
        values = dict(state)
        values["messages"] = [
            {
                "type": message.type,
                "content": message.content,
                "id": message.id,
                "additional_kwargs": message.additional_kwargs,
            }
            for message in state.get("messages", [])
        ]
        return restore_view_state({"checkpoints": [{"values": values}]})

    return SimpleNamespace(
        client=client,
        writer=None,
        resolve_thread=lambda plan: plan.thread_id,
        load_view_state=load,
        recent_threads=lambda **kw: client.list_threads()["thread_list"],
    )


@pytest.mark.asyncio
async def test_tui_mismatch_shows_history_blocks_execution_and_returns(catalog, ui_client, monkeypatch):
    from my_assistant.tui.app import MyAssistantTUI
    from my_assistant.tui.cli import LaunchPlan

    _, alpha, beta = catalog
    ui_client.set_task("original", "finish safely", workspace=alpha.to_context())
    accessor, config, _ = ui_client._workspace_state("original")
    accessor.update(config, {"messages": [HumanMessage("previous history", id="history")]}, as_node="workspace_control")
    app = MyAssistantTUI(_ui_session(ui_client), LaunchPlan(mode="tui", thread_id="original", workspace_id=beta.id))
    submitted = []
    monkeypatch.setattr(app, "run_worker", lambda *a, **kw: submitted.append((a, kw)))
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause()
        assert app._workspace_blocked
        assert app._conv_thread_id == "original"
        assert any(getattr(row, "text", "") == "previous history" for row in app.state.rows)
        app._send_to_agent("try writing")
        app._handle_task("continue")
        assert submitted == []
        app._handle_workspace("return")
        assert app._workspace_context["id"] == alpha.id
        assert app._conv_thread_id == "original" and not app._workspace_blocked
        app._send_to_agent("continue")
        assert len(submitted) == 1
        app._handle_workspace(f"use {beta.id}")
        assert app._workspace_context["id"] == alpha.id  # active worker prevents switching
        app._interrupt_run()
        app._handle_workspace(f"use {beta.id}")
        assert app._conv_thread_id is None
        assert app._workspace_context["id"] == beta.id
        assert not any(getattr(row, "text", "") == "previous history" for row in app.state.rows)


@pytest.mark.asyncio
async def test_tui_startup_picker_waits_for_explicit_selection(catalog, ui_client, monkeypatch):
    from my_assistant.tui.app import MyAssistantTUI, SelectScreen
    from my_assistant.tui.cli import LaunchPlan

    _, alpha, _ = catalog
    app = MyAssistantTUI(_ui_session(ui_client), LaunchPlan(mode="tui", message="initial request"))
    sent = []
    monkeypatch.setattr(app, "_send_to_agent", lambda text, **kw: sent.append((text, app._workspace_context)))
    async with app.run_test(size=(120, 35)) as pilot:
        await pilot.pause()
        assert isinstance(app.screen, SelectScreen)
        assert sent == []
        app.screen.dismiss(alpha.id)
        await pilot.pause()
        assert sent == [("initial request", alpha.to_context())]
        app._handle_workspace("")
        await pilot.pause()
        assert isinstance(app.screen, SelectScreen)
        app.screen.dismiss("__thread_local__")
        await pilot.pause()
        assert app._workspace_context is None


@pytest.mark.asyncio
async def test_tui_policy_remove_and_explicit_task_migration(catalog, ui_client):
    from my_assistant.tui.app import MyAssistantTUI
    from my_assistant.tui.cli import LaunchPlan

    registry, alpha, beta = catalog
    ui_client.set_task("original", "finish alpha", workspace=alpha.to_context())
    app = MyAssistantTUI(_ui_session(ui_client), LaunchPlan(mode="tui", thread_id="original", workspace_id=alpha.id))
    async with app.run_test(size=(120, 35)):
        app._handle_workspace(f'policy {alpha.id} {{"name": "Alpha Project"}}')
        assert app._workspace_context["name"] == "Alpha Project"
        app._handle_workspace('policy "Alpha Project" {"shell_enabled": true}')
        assert app._workspace_context["shell_enabled"] is True
        app._handle_workspace(f'policy {alpha.id} {{"allowed_write_paths": [], "network_enabled": true}}')
        assert app._workspace_context["allowed_write_paths"] == []
        assert app._workspace_context["network_enabled"] is True
        app._handle_workspace(f"remove {alpha.id} confirm")
        assert app._workspace_error
        assert not app._check_workspace_access()
        assert ui_client.get_task("original")["task"]["objective"] == "finish alpha"
        app._handle_workspace(f"migrate {beta.id}")  # missing confirmation is no-op
        assert app._conv_thread_id == "original"
        app._handle_workspace(f"migrate {beta.id} confirm")
        assert app._conv_thread_id != "original"
        assert app._workspace_context["id"] == beta.id
        assert not app._workspace_blocked and app._workspace_error is None
        task = ui_client.get_task(app._conv_thread_id)["task"]
        assert task["status"] == "paused"
        assert task["workspace_binding"] == workspace_binding(beta)
        assert registry.thread_binding("original") == workspace_binding(alpha)


def test_cli_registration_and_selection_reuse_policy(catalog):
    from my_assistant.tui.cli import LaunchPlan, _workspace_context

    registry, alpha, _ = catalog
    context = _workspace_context(LaunchPlan(mode="print", workspace_path=alpha.root_path, workspace_shell=True))
    assert context["id"] == alpha.id and context["shell_enabled"] is True
    assert _workspace_context(LaunchPlan(mode="print", workspace_id=alpha.id)) == context
    assert registry.resolve(alpha.id).shell_enabled


@pytest.mark.parametrize("mode", ["full", "delta"])
def test_task_goal_writes_preserve_binding_and_message_ancestry(catalog, mode):
    from my_assistant.runtime.goal import build_goal_state, write_thread_goal
    from my_assistant.runtime.task import write_thread_task

    _, alpha, _ = catalog
    client = LocalClient(mode=mode)
    binding = client.check_thread_workspace("lineage", workspace=alpha.to_context(), bind=True)
    accessor, config, _ = client._workspace_state("lineage")
    accessor.update(config, {"messages": [HumanMessage("one", id="one")]}, as_node="workspace_control")
    accessor.update(config, {"messages": [HumanMessage("two", id="two")]}, as_node="workspace_control")
    task = {**create_task_state("goal"), "workspace_binding": binding}
    asyncio.run(write_thread_task(client.saver, "lineage", task))
    asyncio.run(write_thread_goal(client.saver, "lineage", build_goal_state("goal"), workspace_binding=binding))
    _, _, state = client._workspace_state("lineage")
    assert [msg.content for msg in state["messages"]] == ["one", "two"]
    assert state["workspace_binding"] == binding == state["task"]["workspace_binding"]
    assert state["goal"]["objective"] == "goal"


@pytest.mark.parametrize("method,args", [
    ("list_file_write_operations", ()),
    ("rollback_file_write_operation", ("operation",)),
    ("undo_latest_file_write", ()),
    ("abandon_file_write_operation", ("operation",)),
])
def test_file_recovery_apis_reject_wrong_workspace_before_io(catalog, ui_client, monkeypatch, method, args):
    _, alpha, beta = catalog
    ui_client.check_thread_workspace("guarded", workspace=alpha.to_context(), bind=True)
    calls = []
    monkeypatch.setattr("my_assistant.sandbox.file_transactions.transaction_root", lambda *a, **kw: calls.append(kw))
    with pytest.raises(WorkspaceMismatch):
        getattr(ui_client, method)("guarded", *args, workspace=beta.to_context())
    assert calls == []


def test_client_rejects_wrong_workspace_before_model_or_tracing(catalog, ui_client, monkeypatch):
    _, alpha, beta = catalog
    ui_client.check_thread_workspace("guarded", workspace=alpha.to_context(), bind=True)
    calls = []
    monkeypatch.setattr(ui_client, "_get_runnable_config", lambda *a, **kw: {"configurable": {"thread_id": "guarded"}})
    monkeypatch.setattr(ui_client, "_ensure_agent", lambda *a, **kw: calls.append("model"))
    with pytest.raises(WorkspaceMismatch):
        list(ui_client._stream_turn("write", thread_id="guarded", workspace=beta.to_context()))
    assert calls == []


def test_late_worker_events_cannot_modify_new_ui_session(catalog, ui_client):
    from my_assistant.tui.app import MyAssistantTUI
    from my_assistant.tui.view_state import SystemMessage

    app = MyAssistantTUI(_ui_session(ui_client), SimpleNamespace(transparent=False, message=None))
    app._run_token = "new-run"
    before = app.state
    app._on_run_action("old-run", SystemMessage("old workspace result"))
    assert app.state == before


def test_workspace_search_indexes_are_independent(catalog):
    from my_assistant.sandbox.workspace_search import WorkspaceFTSIndex

    _, alpha, beta = catalog
    (alpha.root / "notes.md").write_text("alphaonlyphrase", encoding="utf-8")
    (beta.root / "notes.md").write_text("betaonlyphrase", encoding="utf-8")
    a_index, b_index = WorkspaceFTSIndex(alpha), WorkspaceFTSIndex(beta)
    a_hits, _ = a_index.search("alphaonlyphrase", [], max_results=3)
    b_hits, _ = b_index.search("alphaonlyphrase", [], max_results=3)
    assert a_hits and b_hits == []
    assert a_index.db_path != b_index.db_path


def test_re_registering_same_root_does_not_reuse_index(catalog):
    from my_assistant.sandbox.workspace_search import WorkspaceFTSIndex

    registry, alpha, _ = catalog
    old_index = WorkspaceFTSIndex(alpha)
    registry.delete(alpha.id)
    fresh = registry.register(alpha.root)
    assert WorkspaceFTSIndex(fresh).db_path != old_index.db_path


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_agent_graph_persists_workspace_and_blocks_mismatch(catalog, asynchronous):
    from langchain.agents import create_agent
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from my_assistant.agents.thread_state import ThreadState

    _, alpha, beta = catalog
    saver = InMemorySaver()
    model = FakeListChatModel(responses=["done", "should not be called"])
    agent = create_agent(model, middleware=[ThreadDataMiddleware()], state_schema=ThreadState, checkpointer=saver)
    config = {"configurable": {"thread_id": "real-graph"}}
    payload = {"messages": [HumanMessage("inspect")]}
    context = {"thread_id": "real-graph", "workspace": alpha.to_context()}
    if asynchronous:
        state = asyncio.run(agent.ainvoke(payload, config=config, context=context))
    else:
        state = agent.invoke(payload, config=config, context=context)
    assert state["workspace_binding"] == workspace_binding(alpha)
    assert state["thread_data"]["workspace_path"] == alpha.root_path
    assert model.i == 1
    context["workspace"] = beta.to_context()
    with pytest.raises(WorkspaceMismatch):
        if asynchronous:
            asyncio.run(agent.ainvoke(payload, config=config, context=context))
        else:
            agent.invoke(payload, config=config, context=context)
    assert model.i == 1
    assert not (beta.root / ".my-assistant").exists()


@pytest.mark.parametrize("agent_loaded", [False, True])
def test_reference_never_clears_pending_graph_routing(catalog, agent_loaded):
    from langgraph.graph import END, START, StateGraph
    from my_assistant.agents.thread_state import ThreadState

    registry, alpha, beta = catalog
    (alpha.root / "shared.md").write_text("evidence", encoding="utf-8")
    manifest = export_file(registry, alpha.to_context(), "shared.md")
    client = LocalClient()
    binding = client.check_thread_workspace("pending", workspace=beta.to_context(), bind=True)
    builder = StateGraph(ThreadState)
    builder.add_node("waiting", lambda state: {})
    builder.add_edge(START, "waiting")
    builder.add_edge("waiting", END)
    agent = builder.compile(checkpointer=client.saver, interrupt_before=["waiting"])
    config = {"configurable": {"thread_id": "pending"}}
    agent.invoke({"workspace_binding": binding, "messages": [HumanMessage("pending")]}, config)
    before = agent.get_state(config)
    assert before.next == ("waiting",)
    if agent_loaded:
        client._agent = agent
    with pytest.raises(PermissionError, match="routing|interrupted"):
        client.reference_workspace_export("pending", beta.to_context(), manifest["id"])
    assert agent.get_state(config).config == before.config
    assert agent.get_state(config).next == ("waiting",)
    assert not beta.runtime_dir("pending").exists()


def test_export_manifest_listing_survives_restart_and_revoke_blocks_new_references(catalog):
    from my_assistant.workspace_transfers import forget_export, list_exports

    registry, alpha, beta = catalog
    (alpha.root / "export.md").write_text("explicit evidence", encoding="utf-8")
    manifest = export_file(registry, alpha.to_context(), "export.md")
    restarted = WorkspaceRegistry(registry.root)
    assert list_exports(restarted) == [manifest]
    reference_export(restarted, beta.to_context(), "attached", manifest["id"])
    forget_export(restarted, manifest["id"])
    assert list_exports(restarted) == []
    with pytest.raises(LookupError):
        reference_export(restarted, beta.to_context(), "new", manifest["id"])
    assert (beta.runtime_dir("attached") / "uploads" / f"{manifest['id']}.txt").exists()
