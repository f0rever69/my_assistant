from types import SimpleNamespace

import pytest
from my_assistant.tui.app import MyAssistantTUI
from my_assistant.tui.runtime import restore_view_state
from my_assistant.tui.view_state import AssistantRow, ToolRow, UserRow, ViewState


def test_restore_view_state_uses_latest_materialized_checkpoint() -> None:
    payload = {
        "thread_id": "thread-1",
        "checkpoints": [
            {
                "values": {
                    "title": "Old title",
                    "messages": [
                        {"type": "human", "content": "old question", "id": "old-user"}
                    ],
                }
            },
            {
                "values": {
                    "title": "Generated title",
                    "messages": [
                        {
                            "type": "human",
                            "content": "inspect the file",
                            "id": "user-1",
                        },
                        {
                            "type": "ai",
                            "content": "I will inspect it.",
                            "id": "ai-1",
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "name": "read_file",
                                    "args": {"path": "notes.md"},
                                }
                            ],
                        },
                        {
                            "type": "tool",
                            "content": "saved content",
                            "name": "read_file",
                            "tool_call_id": "call-1",
                            "id": "tool-1",
                        },
                        {"type": "ai", "content": "The file is valid.", "id": "ai-2"},
                    ],
                }
            },
        ],
    }

    state = restore_view_state(payload, title_override="Manual title")

    assert state.title == "Manual title"
    assert state.streaming is False
    assert [type(row) for row in state.rows] == [
        UserRow,
        AssistantRow,
        ToolRow,
        AssistantRow,
    ]
    assert state.rows[0].text == "inspect the file"
    assert state.rows[1].id == "ai-1"
    assert state.rows[1].text == "I will inspect it."
    assert state.rows[2].tool_call_id == "call-1"
    assert state.rows[2].status == "ok"
    assert state.rows[2].result == "saved content"
    assert state.rows[3].text == "The file is valid."


def test_restore_view_state_handles_multimodal_text_and_title_fallback() -> None:
    payload = {
        "checkpoints": [
            {
                "values": {
                    "title": "Checkpoint title",
                    "messages": [
                        {
                            "type": "human",
                            "content": [
                                {"type": "text", "text": "describe this"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "data:image/png;base64,..."},
                                },
                            ],
                        },
                        {"type": "system", "content": "hidden agent context"},
                        {
                            "type": "human",
                            "content": "Continue the saved task from its checkpointed next action.",
                            "additional_kwargs": {
                                "hide_from_ui": True,
                                "task_resume_trigger": True,
                            },
                        },
                        {
                            "type": "ai",
                            "content": "hidden internal result",
                            "additional_kwargs": {"hide_from_ui": True},
                        },
                        {
                            "type": "ai",
                            "content": [{"type": "text", "text": "A diagram."}],
                            "id": None,
                        },
                    ],
                }
            }
        ]
    }

    state = restore_view_state(payload)

    assert state.title == "Checkpoint title"
    assert [type(row) for row in state.rows] == [UserRow, AssistantRow]
    assert state.rows[0].text == "describe this"
    assert state.rows[1].text == "A diagram."


def test_restore_view_state_tolerates_empty_or_malformed_history() -> None:
    state = restore_view_state(
        {"checkpoints": [None, {"values": "invalid"}]}, title_override=" Saved "
    )

    assert state.rows == ()
    assert state.title == "Saved"


@pytest.mark.asyncio
async def test_tui_loads_saved_state_when_resuming() -> None:
    restored_state = ViewState(
        rows=(UserRow("previous question"), AssistantRow("previous answer", id="ai-1")),
        title="Saved thread",
    )

    class FakeClient:
        def list_models(self) -> dict:
            return {"models": []}

        def list_skills(self, *, enabled_only: bool) -> dict:
            assert enabled_only is True
            return {"skills": []}

    class FakeSession:
        client = FakeClient()

        def resolve_thread(self, plan) -> str:
            return "thread-1"

        def load_view_state(self, thread_id: str) -> ViewState:
            assert thread_id == "thread-1"
            return restored_state

    plan = SimpleNamespace(message=None, transparent=False)
    app = MyAssistantTUI(FakeSession(), plan)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._conv_thread_id == "thread-1"
        assert app.state == restored_state
