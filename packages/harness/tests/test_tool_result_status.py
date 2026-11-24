from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from my_assistant.agents.middlewares.tool_result_meta import (
    TOOL_META_KEY,
    normalize_tool_result,
)
from my_assistant.tui.runtime import translate


@pytest.mark.parametrize("wrapped", [False, True])
def test_error_results_have_native_error_status(wrapped):
    message = ToolMessage(
        "Error: Unexpected error executing command: sandbox unavailable",
        name="bash",
        tool_call_id="call-1",
    )
    result = Command(update={"messages": [message]}) if wrapped else message
    normalize_tool_result(result, tool_call_id="call-1")
    assert message.additional_kwargs[TOOL_META_KEY]["status"] == "error"
    assert message.status == "error"


@pytest.mark.parametrize("status", ["error", "success"])
def test_preclassified_results_preserve_semantics(status):
    message = ToolMessage(
        "result",
        tool_call_id="call-1",
        additional_kwargs={TOOL_META_KEY: {"status": status}},
    )
    normalize_tool_result(message)
    assert message.status == status


def test_tui_honors_historical_structured_error_metadata():
    data = {
        "type": "tool",
        "name": "bash",
        "tool_call_id": "call-1",
        "status": "success",
        "content": "sandbox unavailable",
        "additional_kwargs": {TOOL_META_KEY: {"status": "error"}},
    }
    [action] = translate(SimpleNamespace(type="messages-tuple", data=data))
    assert action.is_error


def test_tui_preserves_successful_file_content():
    data = {
        "type": "tool",
        "name": "read_file",
        "tool_call_id": "call-1",
        "status": "success",
        "content": "Error: an example error message in a document",
    }
    [action] = translate(SimpleNamespace(type="messages-tuple", data=data))
    assert not action.is_error
