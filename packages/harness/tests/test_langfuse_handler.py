import json
from types import SimpleNamespace

import langfuse
from langchain_core.messages import AIMessage, HumanMessage
from langfuse.langchain import CallbackHandler
from my_assistant.tracing import factory
from my_assistant.tracing.langfuse_handler import (
    MyAssistantLangfuseCallbackHandler,
    mask_langfuse_data,
    summarize_trace_input,
    summarize_trace_output,
)


def test_langfuse_mask_redacts_sensitive_fields_and_common_identifiers() -> None:
    payload = {
        "headers": {"Authorization": "Bearer abc.def.ghi"},
        "apiKey": "secret-api-key",
        "tool_args": '{"api_key": "sk-123456789012345678901234"}',
        "message": "Contact me at user@example.com or 13800138000",
        "private_key": "never-record-this",
    }

    masked = mask_langfuse_data(data=payload)

    assert masked["headers"]["Authorization"] == "[REDACTED]"
    assert masked["apiKey"] == "[REDACTED]"
    assert json.loads(masked["tool_args"]) == {"api_key": "[REDACTED]"}
    assert "user@example.com" not in masked["message"]
    assert "13800138000" not in masked["message"]
    assert masked["private_key"] == "[REDACTED]"
    assert payload["private_key"] == "never-record-this"


def test_langfuse_mask_handles_nested_messages_and_quoted_assignments() -> None:
    masked = mask_langfuse_data(
        data=[
            HumanMessage(content='password="this secret has spaces"'),
            {"auth_token": "do-not-send-either"},
        ]
    )

    assert masked[0]["content"] == 'password="[REDACTED]"'
    assert masked[1]["auth_token"] == "[REDACTED]"


def test_trace_root_input_contains_only_the_latest_user_message() -> None:
    summary = summarize_trace_input(
        {
            "messages": [
                HumanMessage(content="old prompt"),
                AIMessage(content="old response"),
                HumanMessage(content="current prompt"),
            ],
            "summary_text": "private prior conversation summary",
        }
    )

    assert summary == {"user_message": "current prompt"}


def test_trace_root_output_skips_tool_call_messages() -> None:
    summary = summarize_trace_output(
        {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "read_file",
                            "args": {"path": "notes.txt"},
                            "id": "call-1",
                        }
                    ],
                ),
                AIMessage(content="The final answer."),
            ],
            "internal_state": "not part of the trace summary",
        }
    )

    assert summary == {"assistant_response": "The final answer."}


def test_callback_compacts_only_root_inputs_and_outputs(monkeypatch) -> None:
    captured = {"starts": [], "ends": []}

    def capture_start(self, serialized, inputs, **kwargs):
        captured["starts"].append((inputs, kwargs))

    def capture_end(self, outputs, **kwargs):
        captured["ends"].append((outputs, kwargs))

    monkeypatch.setattr(CallbackHandler, "on_chain_start", capture_start)
    monkeypatch.setattr(CallbackHandler, "on_chain_end", capture_end)
    handler = object.__new__(MyAssistantLangfuseCallbackHandler)
    messages = [HumanMessage(content="current question")]
    outputs = {
        "messages": [*messages, AIMessage(content="final response")],
        "other": "omit",
    }

    handler.on_chain_start({}, {"messages": messages}, run_id="root")
    handler.on_chain_start(
        {}, {"messages": messages}, run_id="child", parent_run_id="root"
    )
    handler.on_chain_end(outputs, run_id="root", inputs={"messages": messages})
    handler.on_chain_end(outputs, run_id="child", parent_run_id="root")

    assert captured["starts"][0][0] == {"user_message": "current question"}
    assert captured["starts"][1][0] == {"messages": messages}
    assert captured["ends"][0][0] == {"assistant_response": "final response"}
    assert captured["ends"][0][1]["inputs"] == {"user_message": "current question"}
    assert captured["ends"][1][0] == outputs


def test_callback_error_status_is_masked_without_changing_error_level() -> None:
    handler = object.__new__(MyAssistantLangfuseCallbackHandler)

    level, status = handler._get_error_level_and_status_message(
        RuntimeError("request failed for user@example.com")
    )

    assert level == "ERROR"
    assert status == "request failed for [REDACTED_EMAIL]"


def test_factory_installs_langfuse_mask_and_environment(monkeypatch) -> None:
    captured = {}
    sentinel = object()

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(langfuse, "Langfuse", FakeClient)
    monkeypatch.setattr(
        "my_assistant.tracing.langfuse_handler.MyAssistantLangfuseCallbackHandler",
        lambda *, public_key: sentinel,
    )
    monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", "staging")

    result = factory._create_langfuse_handler(
        SimpleNamespace(
            secret_key="secret",
            public_key="public",
            host="https://example.langfuse.com",
        )
    )

    assert result is sentinel
    assert captured["secret_key"] == "secret"
    assert captured["public_key"] == "public"
    assert captured["base_url"] == "https://example.langfuse.com"
    assert captured["environment"] == "staging"
    assert captured["mask"] is mask_langfuse_data


def test_langfuse_environment_is_normalized_to_sdk_constraints(monkeypatch) -> None:
    monkeypatch.delenv("LANGFUSE_TRACING_ENVIRONMENT", raising=False)
    monkeypatch.setenv("MY_ASSISTANT_ENV", " Local Development ")
    assert factory._resolve_langfuse_environment() == "local-development"

    monkeypatch.setenv("LANGFUSE_TRACING_ENVIRONMENT", "langfuse-internal")
    assert factory._resolve_langfuse_environment() is None
