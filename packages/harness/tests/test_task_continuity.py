from __future__ import annotations

from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from my_assistant.agents.task_continuity.task_state import (
    MAX_TASK_ITEMS,
    create_task_state,
    normalize_task_state,
    update_task_progress,
)
from my_assistant.agents.task_continuity.tools import task_update
from my_assistant.runtime.task import read_thread_task, write_thread_task
from my_assistant.tui.app import MyAssistantTUI
from my_assistant.tui.view_state import ViewState


def test_task_state_is_bounded_deduplicated_and_incremental() -> None:
    task = create_task_state("  finish the project  ")
    updated = update_task_progress(
        task,
        {
            "add_pending_steps": [
                "test",
                "test",
                *[f"step-{i}" for i in range(MAX_TASK_ITEMS + 5)],
            ],
            "add_decisions": ["use sqlite", "use sqlite"],
            "add_modified_files": ["src/app.py"],
            "next_action": "run tests",
        },
    )

    assert updated["objective"] == "finish the project"
    assert len(updated["pending_steps"]) == MAX_TASK_ITEMS
    assert updated["decisions"] == ["use sqlite"]
    assert updated["modified_files"] == ["src/app.py"]
    assert updated["next_action"] == "run tests"
    assert updated["revision"] == 2
    assert (
        update_task_progress(updated, {"status": "completed"})["status"] == "completed"
    )
    with pytest.raises(ValueError):
        update_task_progress(updated, {"status": "abandoned"})

    status_schema = task_update.tool_call_schema.model_json_schema()["properties"][
        "status"
    ]
    assert status_schema["anyOf"][0]["enum"] == ["blocked", "completed"]
    assert "runtime" not in task_update.tool_call_schema.model_fields
    assert normalize_task_state({**updated, "status": "invalid"}) is None


@pytest.mark.asyncio
async def test_task_round_trips_through_checkpoint() -> None:
    saver = InMemorySaver()
    task = update_task_progress(
        create_task_state("resume me"), {"add_pending_steps": ["second step"]}
    )

    await write_thread_task(saver, "thread-task-1", task, create_if_missing=True)

    assert await read_thread_task(saver, "thread-task-1") == task


def _task(status: str = "paused") -> dict:
    return {
        **create_task_state("finish migration"),
        "status": status,
        "pending_steps": ["run tests"],
        "next_action": "run the focused tests",
    }


@pytest.mark.asyncio
async def test_tui_restart_shows_unfinished_task_without_running_it() -> None:
    calls: list[tuple] = []

    class FakeClient:
        def list_models(self):
            return {"models": []}

        def list_skills(self, *, enabled_only: bool):
            return {"skills": []}

        def get_task(self, thread_id: str):
            return {"task": _task()}

    class FakeSession:
        client = FakeClient()

        def resolve_thread(self, plan):
            return "thread-1"

        def load_view_state(self, thread_id: str):
            return ViewState()

    app = MyAssistantTUI(
        FakeSession(), SimpleNamespace(message=None, transparent=False)
    )
    app._send_to_agent = lambda *args, **kwargs: calls.append((args, kwargs))

    async with app.run_test() as pilot:
        await pilot.pause()
        assert calls == []
        assert any(
            "Unfinished task [paused]" in getattr(row, "text", "")
            for row in app.state.rows
        )


@pytest.mark.asyncio
async def test_task_continue_sets_resume_flags_and_prevents_conflict_replay() -> None:
    class FakeClient:
        def __init__(self):
            self.task = _task()
            self.operations = [{"status": "PREPARED"}]
            self.updates = []

        def get_task(self, thread_id):
            return {"task": self.task}

        def list_file_write_operations(self, thread_id, *, workspace=None):
            return self.operations

        def update_task(self, thread_id, patch, *, status=None):
            self.updates.append((patch, status))
            self.task = {**self.task, "status": status or self.task["status"]}
            return {"task": self.task}

        def list_models(self):
            return {"models": []}

        def list_skills(self, *, enabled_only: bool):
            return {"skills": []}

    class FakeSession:
        def __init__(self, client):
            self.client = client

        def resolve_thread(self, plan):
            return "thread-1"

        def load_view_state(self, thread_id):
            return ViewState()

    client = FakeClient()
    app = MyAssistantTUI(
        FakeSession(client),
        SimpleNamespace(transparent=False, workspace=None, message=None),
    )
    sent = []
    app._send_to_agent = lambda text, **kwargs: sent.append((text, kwargs))
    async with app.run_test():
        app.plan = None
        app._handle_task("continue")
        assert sent == [
            (
                "Continue the saved task from its checkpointed next action.",
                {"resume_file_writes": True, "resume_task": True},
            )
        ]
        assert client.updates[-1][1] == "active"

        client.operations = [
            {"status": "CONFLICT", "display_path": "a.txt", "error": "external change"}
        ]
        sent.clear()
        app._handle_task("continue")
        assert sent == []
        assert client.updates[-1][1] == "blocked"
        assert (
            "File recovery conflict" in client.updates[-1][0]["add_failed_attempts"][0]
        )
