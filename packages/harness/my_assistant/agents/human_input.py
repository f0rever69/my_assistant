"""Structured human-input message metadata helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, ToolMessage

HUMAN_INPUT_RESPONSE_KEY = "human_input_response"


class HumanInputTextResponse(TypedDict):
    version: Literal[1]
    kind: Literal["human_input_response"]
    source: str
    request_id: str
    response_kind: Literal["text"]
    value: str


class HumanInputOptionResponse(TypedDict):
    version: Literal[1]
    kind: Literal["human_input_response"]
    source: str
    request_id: str
    response_kind: Literal["option"]
    option_id: str
    value: str


HumanInputResponse = HumanInputTextResponse | HumanInputOptionResponse


def _non_empty_string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def read_human_input_response(additional_kwargs: Mapping[str, object] | None) -> HumanInputResponse | None:
    """Read a valid human-input response payload from message metadata."""
    if not additional_kwargs:
        return None
    raw = additional_kwargs.get(HUMAN_INPUT_RESPONSE_KEY)
    if not isinstance(raw, Mapping):
        return None
    if raw.get("version") != 1 or raw.get("kind") != "human_input_response":
        return None

    source = _non_empty_string(raw.get("source"))
    request_id = _non_empty_string(raw.get("request_id"))
    value = _non_empty_string(raw.get("value"))
    if source is None or request_id is None or value is None:
        return None

    response_kind = raw.get("response_kind")
    if response_kind == "text":
        return {
            "version": 1,
            "kind": "human_input_response",
            "source": source,
            "request_id": request_id,
            "response_kind": "text",
            "value": value,
        }
    if response_kind == "option":
        option_id = _non_empty_string(raw.get("option_id"))
        if option_id is None:
            return None
        return {
            "version": 1,
            "kind": "human_input_response",
            "source": source,
            "request_id": request_id,
            "response_kind": "option",
            "option_id": option_id,
            "value": value,
        }
    return None


def action_confirmation_request_id(messages: object, *, action: str, target: str) -> str | None:
    """Return a fresh approval id bound to one action and target."""
    if not isinstance(messages, list) or not target:
        return None

    latest_human_index = max(
        (index for index, message in enumerate(messages) if isinstance(message, HumanMessage)),
        default=-1,
    )
    if latest_human_index < 0:
        return None

    response_message = messages[latest_human_index]
    response = read_human_input_response(getattr(response_message, "additional_kwargs", None))
    answer = _message_text(getattr(response_message, "content", ""))
    normalized_answer = answer.strip().casefold().rstrip(".!?\u3002\uff01\uff1f")
    explicit_affirmative = normalized_answer in _AFFIRMATIVE_CONFIRMATIONS

    request_index = -1
    request_payload: Mapping[str, object] | None = None
    latest_request_index = -1
    for index, message in enumerate(messages[:latest_human_index]):
        if not isinstance(message, ToolMessage):
            continue
        artifact = message.artifact if isinstance(message.artifact, Mapping) else {}
        payload = artifact.get("human_input")
        if not isinstance(payload, Mapping) or payload.get("kind") != "human_input_request":
            continue
        latest_request_index = index
        if (
            payload.get("source") == "ask_clarification"
            and payload.get("clarification_type") == "risk_confirmation"
            and payload.get("action") == action
            and payload.get("target_path") == target
        ):
            request_index = index
            request_payload = payload

    if request_payload is None or request_index != latest_request_index:
        return None

    first_human_after_request = next(
        (
            index
            for index in range(request_index + 1, len(messages))
            if isinstance(messages[index], HumanMessage)
        ),
        -1,
    )
    if first_human_after_request != latest_human_index:
        return None

    request_id = request_payload.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        return None

    if response is not None:
        if response["source"] != "ask_clarification" or response["request_id"] != request_id:
            return None
        if response["response_kind"] == "option":
            if (
                response.get("option_id") != request_payload.get("approval_option_id")
                or response["value"] != "\u786e\u8ba4\u5220\u9664"
            ):
                return None
        elif response["value"].strip().casefold().rstrip(".!?\u3002\uff01\uff1f") not in _AFFIRMATIVE_CONFIRMATIONS:
            return None
    elif not explicit_affirmative:
        return None

    # A confirmation is single-use, even if the same path is recreated later
    # or the original response is no longer the latest user message.
    for message in messages[request_index + 1 :]:
        if not isinstance(message, ToolMessage) or message.name != action:
            continue
        artifact = message.artifact if isinstance(message.artifact, Mapping) else {}
        consumed = artifact.get("delete_file_confirmation")
        if isinstance(consumed, Mapping) and consumed.get("request_id") == request_id:
            return None

    return request_id


_AFFIRMATIVE_CONFIRMATIONS = frozenset(
    {
        "\u786e\u8ba4\u5220\u9664",
        "\u662f",
        "\u662f\u7684",
        "yes",
        "y",
        "confirm",
        "confirmed",
        "approve",
        "approved",
    }
)


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        item.get("text", "")
        for item in content
        if isinstance(item, Mapping) and isinstance(item.get("text"), str)
    )


def has_action_confirmation(messages: object, *, action: str, target: str) -> bool:
    """Check for a fresh, explicit user approval bound to one action and target."""
    return action_confirmation_request_id(messages, action=action, target=target) is not None
