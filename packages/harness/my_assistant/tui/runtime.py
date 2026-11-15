"""Runtime bridge between ``MyAssistantClient`` streaming and the view-state reducer.

Two layers, both kept free of Textual:

* :func:`translate` — pure: one ``StreamEvent`` -> zero or more reducer actions.
* :func:`stream_actions` — drives ``client.stream()`` and yields a bracketed
  action sequence (``RunStarted`` … translated actions … ``RunEnded``), turning
  model errors into an ``AssistantError`` row instead of crashing.

The Textual app runs :func:`stream_actions` in a worker thread and applies each
yielded action to the reducer on the UI thread.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from .view_state import (
    Action,
    AssistantDelta,
    AssistantError,
    ProgressUpdated,
    RunEnded,
    RunStarted,
    ThreadTitle,
    TodoItem,
    ToolResult,
    ToolStarted,
    UserSubmitted,
    ViewState,
    initial_state,
    reduce,
)


class _StreamEventLike(Protocol):
    type: str
    data: dict


class _ClientLike(Protocol):
    def stream(
        self, message: str, *, thread_id: str | None = None, **kwargs: Any
    ) -> Iterator[Any]:
        """Yield streaming events for *message* (see ``MyAssistantClient.stream``)."""


def translate(event: _StreamEventLike) -> list[Action]:
    """Map a single ``StreamEvent`` to reducer actions. Pure."""
    if event.type == "messages-tuple":
        return _translate_message(event.data)
    if event.type == "end":
        usage = event.data.get("usage") if isinstance(event.data, dict) else None
        actions: list[Action] = []
        error = event.data.get("error") if isinstance(event.data, dict) else None
        if isinstance(error, str) and error.strip():
            actions.append(AssistantError(error.strip()))
        actions.append(RunEnded(usage=usage))
        return actions
    if event.type == "values" and isinstance(event.data, dict):
        actions = []
        title = event.data.get("title")
        if isinstance(title, str) and title.strip():
            actions.append(ThreadTitle(title=title.strip()))
        if "todos" in event.data or "task" in event.data:
            actions.append(project_progress(event.data))
        return actions
    # "custom" events are not rendered incrementally.
    return []


def project_progress(values: dict) -> ProgressUpdated:
    from my_assistant.agents.task_continuity.workflow import normalize_todos

    todos = tuple(TodoItem(**row) for row in normalize_todos(values.get("todos")))
    task = values.get("task")
    return ProgressUpdated(todos=todos, task=task if isinstance(task, dict) else None)


def _translate_message(data: Any) -> list[Action]:
    if not isinstance(data, dict):
        return []

    message_type = data.get("type")
    actions: list[Action] = []

    if message_type == "ai":
        text = _extract_text(data.get("content"))
        if text:
            actions.append(AssistantDelta(id=_as_str(data.get("id")), text=text))
        for tool_call in data.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            actions.append(
                ToolStarted(
                    tool_call_id=_as_str(tool_call.get("id")),
                    tool_name=_as_str(tool_call.get("name")),
                    args=tool_call.get("args") or {},
                )
            )
    elif message_type == "tool":
        is_error = bool(data.get("is_error")) or data.get("status") == "error"
        actions.append(
            ToolResult(
                tool_call_id=_as_str(data.get("tool_call_id")),
                content=_extract_text(data.get("content")),
                is_error=is_error,
                tool_name=_as_str(data.get("name")),
            )
        )

    return actions


def _as_str(value: Any) -> str:
    # Provider stream chunks can carry an explicit ``None`` id/name (the key is
    # present, so ``.get(k, "")`` would return None, and ``str(None) == "None"``
    # — a truthy value that would defeat the empty-id guard downstream).
    return "" if value is None else str(value)


def stream_actions(
    client: _ClientLike, message: str, *, thread_id: str | None = None, **kwargs: Any
) -> Iterator[Action]:
    """Yield a bracketed action stream for one agent run.

    Always begins with ``RunStarted`` and ends with ``RunEnded`` (even on error,
    where an ``AssistantError`` row is emitted first).
    """
    yield RunStarted()
    try:
        for event in client.stream(message, thread_id=thread_id, **kwargs):
            yield from translate(event)
            if event.type == "end":
                return  # RunEnded already emitted by translate()
        yield RunEnded()
    except Exception as exc:  # noqa: BLE001 - surface any model/runtime error in-UI
        yield AssistantError(str(exc) or exc.__class__.__name__)
        yield RunEnded()


def restore_view_state(payload: Any, *, title_override: str | None = None) -> ViewState:
    """Rebuild a non-streaming TUI transcript from a thread history payload.

    ``MyAssistantClient.get_thread()`` returns materialized checkpoints in
    chronological order. The latest checkpoint already contains the complete
    message channel, so replaying older checkpoints would duplicate messages.
    Reusing the live-stream reducer here keeps restored tool cards and message
    formatting consistent with a conversation rendered in real time.
    """
    state = initial_state()
    latest_values: dict[str, Any] = {}

    if isinstance(payload, dict):
        checkpoints = payload.get("checkpoints")
        if isinstance(checkpoints, list):
            for checkpoint in reversed(checkpoints):
                if not isinstance(checkpoint, dict):
                    continue
                values = checkpoint.get("values")
                if isinstance(values, dict):
                    latest_values = values
                    break

    messages = latest_values.get("messages")
    if isinstance(messages, (list, tuple)):
        for message in messages:
            if not isinstance(message, dict):
                continue
            additional_kwargs = message.get("additional_kwargs")
            if (
                isinstance(additional_kwargs, dict)
                and additional_kwargs.get("hide_from_ui") is True
            ):
                continue
            message_type = message.get("type")
            if message_type in {"human", "user"}:
                text = _extract_text(message.get("content"))
                if text:
                    state = reduce(state, UserSubmitted(text=text))
                continue
            if message_type in {"ai", "assistant"}:
                normalized = dict(message)
                normalized["type"] = "ai"
                actions = _translate_message(normalized)
            elif message_type == "tool":
                actions = _translate_message(message)
            else:
                # System/runtime messages are agent context, not user-visible
                # transcript rows. Unknown message types are ignored safely.
                continue
            for action in actions:
                state = reduce(state, action)

    title = title_override.strip() if isinstance(title_override, str) else ""
    if not title:
        checkpoint_title = latest_values.get("title")
        title = checkpoint_title.strip() if isinstance(checkpoint_title, str) else ""
    if title:
        state = reduce(state, ThreadTitle(title=title))
    if "todos" in latest_values or "task" in latest_values:
        state = reduce(state, project_progress(latest_values))
    return state


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)
