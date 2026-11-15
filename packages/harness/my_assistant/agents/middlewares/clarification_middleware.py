"""Middleware for intercepting clarification requests and presenting them to the user."""

import json
import logging
import re
from collections.abc import Callable
from hashlib import sha256
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from my_assistant.agents.human_input import (
    action_confirmation_request_id,
    has_action_confirmation,
)
from my_assistant.agents.middlewares.tool_call_metadata import (
    clone_ai_message_with_tool_calls,
)

logger = logging.getLogger(__name__)

ASK_CLARIFICATION_TOOL_NAME = "ask_clarification"
DELETE_FILE_TOOL_NAME = "delete_file"

# Whitelisted form field types; anything else degrades to "text" so a bad
# model-provided type can never produce an unrenderable card.
FORM_FIELD_TYPES = frozenset({"text", "textarea", "number", "select", "multi_select", "checkbox", "date"})
_OPTION_FIELD_TYPES = frozenset({"select", "multi_select"})

# Field names that collide with JavaScript Object.prototype properties. The
# frontend stores form values in a plain object keyed by field name, so these
# would read inherited prototype members instead of user input.
_RESERVED_FIELD_NAMES = frozenset(
    {
        "__proto__",
        "constructor",
        "prototype",
        "toString",
        "toLocaleString",
        "valueOf",
        "hasOwnProperty",
        "isPrototypeOf",
        "propertyIsEnumerable",
        "__defineGetter__",
        "__defineSetter__",
        "__lookupGetter__",
        "__lookupSetter__",
    }
)

# Hard caps so a runaway model cannot publish an unbounded form. Exceeding a
# cap is a structural error: the whole form degrades to the legacy modes
# instead of silently truncating business fields.
MAX_FORM_FIELDS = 16
MAX_FIELD_OPTIONS = 24
MAX_FIELD_TEXT_CHARS = 200
# Total budget over the serialized normalized fields, in UTF-8 bytes. The
# per-item caps alone still admit forms whose plain-text IM fallback exceeds
# channel delivery limits (Slack truncates at 40k chars per message; Feishu
# guides ~30KB per card), which would silently drop trailing fields — the very
# thing atomic validation exists to prevent. 16KB keeps the fallback text of
# any accepted form comfortably inside the strictest supported channel while
# leaving headroom for question/context.
MAX_FORM_SERIALIZED_BYTES = 16_384

_XML_TAG_RE = re.compile(r"</?[A-Za-z_][\w:.-]*(?:\s[^<>]*?)?\s*/?>")


class ClarificationMiddlewareState(AgentState):
    """Compatible with the `ThreadState` schema."""

    pass


class ClarificationMiddleware(AgentMiddleware[ClarificationMiddlewareState]):
    """Intercepts clarification tool calls and interrupts execution to present questions to the user.

    When the model calls `ask_clarification`, this middleware:
    1. Drops any sibling tool calls from the same AIMessage (``after_model``)
       so they cannot execute before the user answers. langchain's
       ``return_direct`` router inspects all client-side tool calls of the
       last AIMessage and routes to END only when every one is
       ``return_direct``. A mixed ``[ask_clarification, bash]`` batch would
       both run the siblings *and* loop back to the model. Malformed
       ``ask_clarification`` arguments land in ``invalid_tool_calls`` while a
       valid sibling stays in ``tool_calls``; that still counts as a stop
       signal and the siblings are dropped.
    2. Intercepts the remaining ``ask_clarification`` call before execution
    3. Extracts the clarification question and metadata
    4. Formats a user-friendly message
    5. Returns a Command that interrupts execution and presents the question
    6. Waits for user response before continuing

    Unconfirmed `delete_file` calls are also converted into path-bound risk
    confirmations and cannot reach the tool implementation until the user
    approves that exact request.
    """

    state_schema = ClarificationMiddlewareState

    def _stable_message_id(self, tool_call_id: str, formatted_message: str) -> str:
        """Build a deterministic message ID so retried clarification calls replace, not append."""
        if tool_call_id:
            return f"clarification:{tool_call_id}"
        digest = sha256(formatted_message.encode("utf-8")).hexdigest()[:16]
        return f"clarification:{digest}"

    def _normalize_options(self, raw_options: Any) -> list[str]:
        """Normalize tool-provided options into displayable string values."""
        options = raw_options

        # Some models (e.g. Qwen3-Max) serialize array parameters as JSON strings
        # instead of native arrays. Deserialize and normalize so `options`
        # is always a list for the rendering logic below.
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except (json.JSONDecodeError, TypeError):
                options = [options]

        if options is None:
            return []
        if isinstance(options, dict):
            options = self._flatten_dict_option_values(options)
        elif not isinstance(options, list):
            options = [options]

        # Trim, drop blanks, and dedupe (order-preserving): the frontend parser
        # rejects the whole payload on blank option labels, so they must never
        # be emitted.
        normalized: list[str] = []
        seen: set[str] = set()
        for option in options:
            text = _XML_TAG_RE.sub("", str(option)).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized.append(text)
        return normalized

    @staticmethod
    def _flatten_dict_option_values(value: dict[str, Any]) -> list[str | int | float]:
        """Flatten scalar leaves from XML-to-dict option payloads in source order."""
        flattened: list[str | int | float] = []

        def collect(nested: Any) -> None:
            if isinstance(nested, dict):
                for item in nested.values():
                    collect(item)
            elif isinstance(nested, list):
                for item in nested:
                    collect(item)
            elif isinstance(nested, str | int | float):
                flattened.append(nested)

        collect(value)
        return flattened

    @staticmethod
    def _normalize_bool(raw: Any) -> bool:
        """Coerce a model-provided boolean; some models serialize booleans as strings or 1/0."""
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, int | float):
            return bool(raw)
        if isinstance(raw, str):
            return raw.strip().lower() == "true"
        return False

    def _normalize_fields(self, raw_fields: Any) -> list[dict[str, Any]]:
        """Normalize tool-provided form fields into the validated v2 field schema.

        Validation is atomic: any structurally broken entry (non-dict, bad or
        reserved or duplicate name, over-cap counts/lengths) invalidates the
        whole form so the card can never render "complete" while silently
        missing a required business field. Benign issues keep their local
        degradation: unknown types and option-less selects become ``text``.
        """
        fields = raw_fields
        if isinstance(fields, str):
            try:
                fields = json.loads(fields)
            except (json.JSONDecodeError, TypeError):
                return []
        if not isinstance(fields, list):
            return []
        if len(fields) > MAX_FORM_FIELDS:
            return []

        normalized: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for entry in fields:
            if not isinstance(entry, dict):
                return []
            raw_name = entry.get("name")
            if not isinstance(raw_name, str) or not raw_name.strip():
                return []
            name = raw_name.strip()
            if name in _RESERVED_FIELD_NAMES or name in seen_names or len(name) > MAX_FIELD_TEXT_CHARS:
                return []
            seen_names.add(name)

            raw_label = entry.get("label")
            label = raw_label.strip() if isinstance(raw_label, str) and raw_label.strip() else name
            if len(label) > MAX_FIELD_TEXT_CHARS:
                return []

            field_type = entry.get("type")
            # isinstance guard first: `type: []` / `type: {}` are legal JSON
            # from a model, and an unhashable membership probe would raise
            # TypeError instead of degrading.
            if not isinstance(field_type, str) or field_type not in FORM_FIELD_TYPES:
                field_type = "text"

            options = self._normalize_options(entry.get("options")) if field_type in _OPTION_FIELD_TYPES else []
            if len(options) > MAX_FIELD_OPTIONS or any(len(option) > MAX_FIELD_TEXT_CHARS for option in options):
                return []
            if field_type in _OPTION_FIELD_TYPES and not options:
                field_type = "text"

            field: dict[str, Any] = {
                "name": name,
                "label": label,
                "type": field_type,
                "required": self._normalize_bool(entry.get("required")),
            }
            if field_type in _OPTION_FIELD_TYPES:
                field["options"] = [
                    {
                        "id": f"{name}-option-{index}",
                        "label": option,
                        "value": option,
                    }
                    for index, option in enumerate(options, 1)
                ]
            placeholder = entry.get("placeholder")
            if isinstance(placeholder, str) and placeholder.strip():
                if len(placeholder.strip()) > MAX_FIELD_TEXT_CHARS:
                    return []
                field["placeholder"] = placeholder.strip()
            normalized.append(field)

        if len(json.dumps(normalized, ensure_ascii=False).encode("utf-8")) > MAX_FORM_SERIALIZED_BYTES:
            return []

        return normalized

    def _build_human_input_payload(self, args: dict[str, Any], *, tool_call_id: str, request_id: str, fields: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Build the structured UI payload while keeping ToolMessage.content as fallback.

        Protocol versioning: legacy modes (``free_text`` / ``choice_with_other``)
        keep ``version: 1`` so their wire format is unchanged; the v2 ``form``
        mode carries ``version: 2`` so older frontends reject the payload and
        degrade to the plain-text ToolMessage content. Replies stay on the v1
        response protocol (``text`` / ``option``) — the form card submits a
        readable ``value`` summary, so no new response kind is introduced.

        ``fields`` accepts an already-normalized list so callers rendering both
        the payload and the text fallback normalize only once.
        """
        if fields is None:
            fields = self._normalize_fields(args.get("fields"))
        options = self._normalize_options(args.get("options", []))
        clarification_type = str(args.get("clarification_type", "missing_info"))

        if fields:
            version, input_mode = 2, "form"
        elif options:
            version, input_mode = 1, "choice_with_other"
        else:
            version, input_mode = 1, "free_text"

        payload: dict[str, Any] = {
            "version": version,
            "kind": "human_input_request",
            "source": "ask_clarification",
            "request_id": request_id,
            "clarification_type": clarification_type,
            "question": str(args.get("question") or ""),
            "input_mode": input_mode,
        }

        if tool_call_id:
            payload["tool_call_id"] = tool_call_id

        if "context" in args:
            context = args.get("context")
            payload["context"] = None if context is None else str(context)

        if input_mode == "form":
            payload["fields"] = fields
        elif options:
            payload["options"] = [
                {
                    "id": f"option-{index}",
                    "label": option,
                    "value": option,
                }
                for index, option in enumerate(options, 1)
            ]

        return payload

    def _is_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters.

        Args:
            text: Text to check

        Returns:
            True if text contains Chinese characters
        """
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _format_clarification_message(self, args: dict, fields: list[dict[str, Any]] | None = None) -> str:
        """Format the clarification arguments into a user-friendly message.

        Args:
            args: The tool call arguments containing clarification details
            fields: Already-normalized form fields, so callers rendering both
                the payload and this fallback normalize only once

        Returns:
            Formatted message string
        """
        question = args.get("question", "")
        # str() coercion keeps the icon lookup hashable — `clarification_type:
        # []` is legal JSON from a model and would raise TypeError as a dict key.
        clarification_type = str(args.get("clarification_type", "missing_info"))
        context = args.get("context")
        if fields is None:
            fields = self._normalize_fields(args.get("fields"))
        options = self._normalize_options(args.get("options", []))

        # Type-specific icons
        type_icons = {
            "missing_info": "❓",
            "ambiguous_requirement": "🤔",
            "approach_choice": "🔀",
            "risk_confirmation": "⚠️",
            "suggestion": "💡",
        }

        icon = type_icons.get(clarification_type, "❓")

        # Build the message naturally
        message_parts = []

        # Add icon and question together for a more natural flow
        if context:
            # If there's context, present it first as background
            message_parts.append(f"{icon} {context}")
            message_parts.append(f"\n{question}")
        else:
            # Just the question with icon
            message_parts.append(f"{icon} {question}")

        # Form fields take precedence over options, mirroring the payload logic.
        if fields:
            message_parts.append("")  # blank line for spacing
            for i, field in enumerate(fields, 1):
                line = f"  {i}. {field['label']}"
                if field["required"]:
                    line += " (required)"
                field_options = field.get("options")
                if field_options:
                    line += " — options: " + " / ".join(option["label"] for option in field_options)
                    if field["type"] == "multi_select":
                        line += " (multiple allowed)"
                message_parts.append(line)
            message_parts.append("")
            message_parts.append("Please reply with a value for each field.")
        elif options and len(options) > 0:
            message_parts.append("")  # blank line for spacing
            for i, option in enumerate(options, 1):
                message_parts.append(f"  {i}. {option}")

        return "\n".join(message_parts)

    def _clarification_disabled(self, runtime: Any) -> bool:
        """Whether clarifications are suppressed for this run.

        Non-interactive channels (e.g. GitHub webhooks) set
        ``disable_clarification`` in the run context because a clarification
        would dead-end the run — the human only "replies" via a later
        webhook delivery, by which point the agent's turn is long over.
        """
        context = getattr(runtime, "context", None)
        if not context:
            return False
        return bool(context.get("disable_clarification"))

    def _is_disabled(self, request: ToolCallRequest) -> bool:
        """Whether clarifications are suppressed for this tool-call request."""
        return self._clarification_disabled(getattr(request, "runtime", None))

    def _drop_parallel_non_clarification_tools(self, state: AgentState, runtime: Runtime) -> dict | None:
        """Keep only a clarification or pending deletion when tools were batched.

        Providers routinely batch tool calls. If ``ask_clarification`` shares a
        turn with ``bash`` / ``write_file`` / ..., those siblings execute before
        the user answers, and langchain's ``return_direct`` check (every
        client-side tool call of the last AIMessage must be ``return_direct``)
        routes back to the model. Rewrite the AIMessage so the tools node
        never sees the siblings.

        LangChain parses each provider call independently, so a malformed
        ``ask_clarification`` is stored on ``invalid_tool_calls`` while a
        valid sibling remains executable on ``tool_calls``. Treat that
        malformed call as the same stop signal: drop the siblings so the
        tools node cannot run them. With no remaining ``tool_calls``,
        ``create_agent`` routes to END.

        ``disable_clarification`` keeps ordinary clarification siblings because
        those calls become a "proceed" ToolMessage. Pending deletions remain
        isolated and blocked because there is no human available to approve them.
        """
        messages = state.get("messages", [])
        if not messages:
            return None
        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None

        tool_calls = list(last.tool_calls or [])
        invalid_tool_calls = [tc for tc in (getattr(last, "invalid_tool_calls", None) or []) if isinstance(tc, dict)]
        clarification_calls = [tc for tc in tool_calls if tc.get("name") == ASK_CLARIFICATION_TOOL_NAME]
        invalid_clarification_calls = [tc for tc in invalid_tool_calls if tc.get("name") == ASK_CLARIFICATION_TOOL_NAME]
        if (clarification_calls or invalid_clarification_calls) and not self._clarification_disabled(runtime):
            sibling_calls = [tc for tc in tool_calls if tc.get("name") != ASK_CLARIFICATION_TOOL_NAME]
            if sibling_calls:
                dropped_names = [str(tc.get("name") or "unknown") for tc in sibling_calls]
                logger.warning(
                    "ask_clarification was emitted with %d sibling tool call(s); dropping %s so they cannot run before the user answers",
                    len(dropped_names),
                    dropped_names,
                )
                return {"messages": [clone_ai_message_with_tool_calls(last, clarification_calls)]}
            return None

        unconfirmed_deletions = [
            call
            for call in tool_calls
            if call.get("name") == DELETE_FILE_TOOL_NAME
            and not has_action_confirmation(
                messages,
                action=DELETE_FILE_TOOL_NAME,
                target=str((call.get("args") or {}).get("path") or "") if isinstance(call.get("args"), dict) else "",
            )
        ]
        if not unconfirmed_deletions:
            return None

        kept_call = unconfirmed_deletions[0]
        dropped_names = [
            str(tc.get("name") or "unknown")
            for tc in tool_calls
            if tc is not kept_call
        ]
        logger.warning(
            "Unconfirmed delete_file call emitted with %d sibling tool call(s); dropping %s until the deletion is confirmed",
            len(dropped_names),
            dropped_names or ["duplicate delete_file"],
        )
        return {"messages": [clone_ai_message_with_tool_calls(last, [kept_call])]}

    def _handle_delete_confirmation(self, request: ToolCallRequest) -> Command | ToolMessage:
        """Pause a deletion call and bind the user response to its exact path."""
        args = request.tool_call.get("args", {})
        path = args.get("path") if isinstance(args, dict) else None
        tool_call_id = str(request.tool_call.get("id") or "")
        if not isinstance(path, str) or not path or len(path) > 4096 or "\x00" in path:
            return ToolMessage(
                content="Error: delete_file requires a valid file path; no file was deleted.",
                tool_call_id=tool_call_id,
                name=DELETE_FILE_TOOL_NAME,
            )
        if self._clarification_disabled(getattr(request, "runtime", None)):
            return ToolMessage(
                content="Error: file deletion requires an interactive user confirmation; no file was deleted.",
                tool_call_id=tool_call_id,
                name=DELETE_FILE_TOOL_NAME,
            )

        display_path = json.dumps(path, ensure_ascii=False)
        description = args.get("description") if isinstance(args, dict) else None
        description = " ".join(description.split())[:300] if isinstance(description, str) else ""
        context = "删除文件不可撤销，请核对目标路径。"
        if description:
            context += f" 操作说明：{description}"
        clarification_args = {
            "question": f"确认删除文件 {display_path} 吗？",
            "clarification_type": "risk_confirmation",
            "context": context,
            "options": ["确认删除", "取消"],
        }
        question = self._format_clarification_message(clarification_args)
        request_id = self._stable_message_id(tool_call_id, question)
        payload = self._build_human_input_payload(
            clarification_args,
            tool_call_id=tool_call_id,
            request_id=request_id,
        )
        payload.update(
            {
                "action": DELETE_FILE_TOOL_NAME,
                "target_path": path,
                "approval_option_id": "option-1",
            }
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        id=request_id,
                        content=question,
                        tool_call_id=tool_call_id,
                        name=DELETE_FILE_TOOL_NAME,
                        artifact={"human_input": payload},
                    )
                ]
            },
            goto=END,
        )

    @staticmethod
    def _stamp_delete_confirmation(
        result: ToolMessage | Command,
        *,
        path: str,
        messages: object,
    ) -> ToolMessage | Command:
        if not isinstance(result, ToolMessage):
            return result
        request_id = action_confirmation_request_id(
            messages,
            action=DELETE_FILE_TOOL_NAME,
            target=path,
        )
        if not request_id:
            return result
        artifact = dict(result.artifact) if isinstance(result.artifact, dict) else {}
        artifact["delete_file_confirmation"] = {"request_id": request_id, "target_path": path}
        return result.model_copy(update={"artifact": artifact})

    def _handle_disabled_clarification(self, request: ToolCallRequest) -> ToolMessage:
        """Suppress a clarification and tell the agent to proceed.

        Returns a plain ToolMessage (not a ``Command(goto=END)``) so the
        agent loop continues instead of ending — the agent receives this
        as the tool result and generates again, ideally acting rather
        than re-asking.
        """
        tool_call_id = request.tool_call.get("id", "")
        logger.info("ask_clarification suppressed (disable_clarification set); instructing agent to proceed")
        return ToolMessage(
            id=self._stable_message_id(tool_call_id, "proceed-without-clarification"),
            content=(
                "Clarification is disabled in this context — the human is not present "
                "to answer synchronously. Do not ask for confirmation. Proceed with your "
                "best judgment, carry out the requested action, and state any assumptions "
                "you made in your final response."
            ),
            tool_call_id=tool_call_id,
            name=ASK_CLARIFICATION_TOOL_NAME,
        )

    def _handle_clarification(self, request: ToolCallRequest) -> Command:
        """Handle clarification request and return command to interrupt execution.

        Args:
            request: Tool call request

        Returns:
            Command that interrupts execution with the formatted clarification message
        """
        # Extract clarification arguments
        args = request.tool_call.get("args", {})
        question = args.get("question", "")

        logger.info("Intercepted clarification request")
        logger.debug("Clarification question: %s", question)

        # Normalize form fields once; both the text fallback and the payload
        # consume the same result.
        fields = self._normalize_fields(args.get("fields"))

        # Format the clarification message
        formatted_message = self._format_clarification_message(args, fields=fields)

        # Get the tool call ID
        tool_call_id = request.tool_call.get("id", "")

        request_id = self._stable_message_id(tool_call_id, formatted_message)
        human_input_payload = self._build_human_input_payload(args, tool_call_id=tool_call_id, request_id=request_id, fields=fields)

        # Create a ToolMessage with the formatted question
        # This will be added to the message history
        tool_message = ToolMessage(
            id=request_id,
            content=formatted_message,
            tool_call_id=tool_call_id,
            name=ASK_CLARIFICATION_TOOL_NAME,
            artifact={"human_input": human_input_payload},
        )

        # Return a Command that:
        # 1. Adds the formatted tool message
        # 2. Interrupts execution by going to __end__
        # Note: We don't add an extra AIMessage here - the frontend will detect
        # and display ask_clarification tool messages directly
        return Command(
            update={"messages": [tool_message]},
            goto=END,
        )

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept ask_clarification tool calls and interrupt execution (sync version).

        Args:
            request: Tool call request
            handler: Original tool execution handler

        Returns:
            Command that interrupts execution with the formatted clarification message
        """
        tool_name = request.tool_call.get("name")
        if tool_name == DELETE_FILE_TOOL_NAME:
            args = request.tool_call.get("args", {})
            path = args.get("path") if isinstance(args, dict) else None
            messages = (request.state or {}).get("messages")
            if isinstance(path, str) and has_action_confirmation(
                messages,
                action=DELETE_FILE_TOOL_NAME,
                target=path,
            ):
                result = handler(request)
                return self._stamp_delete_confirmation(result, path=path, messages=messages)
            return self._handle_delete_confirmation(request)

        if tool_name != ASK_CLARIFICATION_TOOL_NAME:
            return handler(request)

        if self._is_disabled(request):
            return self._handle_disabled_clarification(request)

        return self._handle_clarification(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Intercept ask_clarification tool calls and interrupt execution (async version).

        Args:
            request: Tool call request
            handler: Original tool execution handler (async)

        Returns:
            Command that interrupts execution with the formatted clarification message
        """
        tool_name = request.tool_call.get("name")
        if tool_name == DELETE_FILE_TOOL_NAME:
            args = request.tool_call.get("args", {})
            path = args.get("path") if isinstance(args, dict) else None
            messages = (request.state or {}).get("messages")
            if isinstance(path, str) and has_action_confirmation(
                messages,
                action=DELETE_FILE_TOOL_NAME,
                target=path,
            ):
                result = await handler(request)
                return self._stamp_delete_confirmation(result, path=path, messages=messages)
            return self._handle_delete_confirmation(request)

        if tool_name != ASK_CLARIFICATION_TOOL_NAME:
            return await handler(request)

        if self._is_disabled(request):
            return self._handle_disabled_clarification(request)

        return self._handle_clarification(request)

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._drop_parallel_non_clarification_tools(state, runtime)

    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        return self._drop_parallel_non_clarification_tools(state, runtime)
