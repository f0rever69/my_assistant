"""Privacy-aware Langfuse callback support for MyAssistant graph runs."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import BaseMessage
from langfuse.langchain import CallbackHandler

_REDACTED = "[REDACTED]"
_SENSITIVE_KEY = re.compile(
    r"(?:^|_)(?:access_token|refresh_token|token|api_key|apikey|secret|password|passwd|"
    r"authorization|credential|private_key|client_secret|cookie)(?:_|$)",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)([\"']?(?:access[_-]?token|refresh[_-]?token|token|api[_-]?key|secret|"
    r"password|passwd|authorization|credential|private[_-]?key|client[_-]?secret|"
    r"cookie)[\"']?\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;}]+)"
)
_BEARER_TOKEN = re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/-]+=*")
_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)
_KNOWN_TOKEN = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"AKIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{30,})\b"
)
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_CN_MOBILE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")


def _redact_assignment(match: re.Match[str]) -> str:
    value = match.group(2)
    quote = value[0] if value.startswith(('"', "'")) else ""
    return f"{match.group(1)}{quote}{_REDACTED}{quote}"


def _is_sensitive_key(key: str) -> bool:
    snake_case = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).replace("-", "_")
    return _SENSITIVE_KEY.search(snake_case) is not None


def mask_langfuse_data(*, data: Any) -> Any:
    """Recursively redact credentials and common personal identifiers.

    This function is passed to Langfuse's SDK-level ``mask`` hook, which applies
    before observation input, output, and metadata are recorded.
    """
    if isinstance(data, str):
        masked = _PRIVATE_KEY.sub(_REDACTED, data)
        masked = _SECRET_ASSIGNMENT.sub(_redact_assignment, masked)
        masked = _BEARER_TOKEN.sub(r"\1" + _REDACTED, masked)
        masked = _KNOWN_TOKEN.sub(_REDACTED, masked)
        masked = _EMAIL.sub("[REDACTED_EMAIL]", masked)
        return _CN_MOBILE.sub("[REDACTED_PHONE]", masked)

    if isinstance(data, Mapping):
        masked: dict[Any, Any] = {}
        for key, value in data.items():
            if isinstance(key, str) and _is_sensitive_key(key):
                masked[key] = _REDACTED
            else:
                masked[key] = mask_langfuse_data(data=value)
        return masked

    if isinstance(data, list):
        return [mask_langfuse_data(data=item) for item in data]
    if isinstance(data, tuple):
        return tuple(mask_langfuse_data(data=item) for item in data)

    model_dump = getattr(data, "model_dump", None)
    if callable(model_dump):
        try:
            return mask_langfuse_data(data=model_dump(mode="python"))
        except Exception:  # noqa: BLE001 - fail closed if an object cannot be serialized
            return f"<{type(data).__name__}>"

    if data is None or isinstance(data, (bool, int, float)):
        return data
    return f"<{type(data).__name__}>"


def _message_role_and_content(message: Any) -> tuple[str | None, Any, bool]:
    if isinstance(message, BaseMessage):
        role = message.type
        content = message.content
        tool_calls = bool(getattr(message, "tool_calls", None))
        return role, content, tool_calls
    if isinstance(message, Mapping):
        role = message.get("role") or message.get("type")
        return (
            role if isinstance(role, str) else None,
            message.get("content"),
            bool(message.get("tool_calls")),
        )
    return None, None, False


def _messages_from(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        messages = value.get("messages")
        if isinstance(messages, (list, tuple)):
            return list(messages)
        direct = value.get("input") or value.get("output")
        if isinstance(direct, (BaseMessage, Mapping)):
            return [direct]
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, BaseMessage):
        return [value]
    return []


def summarize_trace_input(inputs: Any) -> dict[str, Any]:
    """Keep only the latest user message on the root trace."""
    if isinstance(inputs, str):
        return {"user_message": inputs}

    for message in reversed(_messages_from(inputs)):
        role, content, _ = _message_role_and_content(message)
        if role in {"human", "user"}:
            return {"user_message": content}

    if isinstance(inputs, Mapping) and isinstance(inputs.get("input"), str):
        return {"user_message": inputs["input"]}
    return {"user_message": "<unavailable>"}


def summarize_trace_output(outputs: Any) -> dict[str, Any]:
    """Keep only the final assistant response on the root trace."""
    if isinstance(outputs, str):
        return {"assistant_response": outputs}

    for message in reversed(_messages_from(outputs)):
        role, content, has_tool_calls = _message_role_and_content(message)
        if role in {"ai", "assistant"} and not has_tool_calls:
            return {"assistant_response": content}

    if isinstance(outputs, Mapping) and isinstance(outputs.get("output"), str):
        return {"assistant_response": outputs["output"]}
    return {"assistant_response": "<unavailable>"}


class MyAssistantLangfuseCallbackHandler(CallbackHandler):
    """Langfuse callback that keeps the graph root focused on one chat turn."""

    def _get_error_level_and_status_message(
        self, error: BaseException
    ) -> tuple[str, str]:
        level, status_message = super()._get_error_level_and_status_message(error)
        return level, mask_langfuse_data(data=status_message)

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            inputs = summarize_trace_input(inputs)
        return super().on_chain_start(
            serialized,
            inputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None:
            outputs = summarize_trace_output(outputs)
            if "inputs" in kwargs:
                kwargs["inputs"] = summarize_trace_input(kwargs["inputs"])
        return super().on_chain_end(
            outputs,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        **kwargs: Any,
    ) -> Any:
        if parent_run_id is None and "inputs" in kwargs:
            kwargs["inputs"] = summarize_trace_input(kwargs["inputs"])
        return super().on_chain_error(
            error,
            run_id=run_id,
            parent_run_id=parent_run_id,
            **kwargs,
        )
