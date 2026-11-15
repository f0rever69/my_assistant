"""Reconcile file side effects with the next persisted thread checkpoint."""

from __future__ import annotations

import logging
from typing import NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from my_assistant.runtime.user_context import resolve_runtime_user_id
from my_assistant.sandbox.file_transactions import (
    ABANDONED,
    ACTIVE_STATUSES,
    APPLIED,
    CONFLICT,
    PREPARED,
    RECOVERY_STATUSES,
    ROLLBACK_PREPARED,
    ROLLED_BACK,
    FileTransactionStore,
    transaction_root,
)

logger = logging.getLogger(__name__)


class FileTransactionState(AgentState):
    thread_data: NotRequired[dict | None]


class FileTransactionRecoveryMiddleware(AgentMiddleware[FileTransactionState]):
    """Recover APPLIED writes and optionally resume PREPARED writes.

    An APPLIED transaction is converted into its missing ToolMessage before a
    later user message reaches the model. The next invocation observes that
    checkpointed message and advances APPLIED -> COMMITTED.
    """

    state_schema = FileTransactionState

    @override
    def before_agent(self, state: FileTransactionState, runtime: Runtime) -> dict | None:
        context = runtime.context if isinstance(runtime.context, dict) else {}
        phase = ((state.get("task") or {}).get("execution") or {}).get("phase")
        if context.get("plan_preview") or phase in {"preview", "awaiting_approval", "blocked", "paused", "abandoned", "delivered", "completed"}:
            # Recovery can itself replay a write or finish a rollback. A
            # read-only planning turn must not acquire those side effects.
            return None
        thread_id = context.get("thread_id") or (runtime.config or {}).get("configurable", {}).get("thread_id")
        if not isinstance(thread_id, str) or not thread_id:
            return None

        store = FileTransactionStore(
            transaction_root(
                thread_id,
                user_id=resolve_runtime_user_id(runtime),
                workspace=context.get("workspace"),
                thread_data=state.get("thread_data"),
            ),
            thread_id,
            workspace=context.get("workspace") or (state.get("thread_data") or {}).get("workspace_policy"),
        )
        messages = list(state.get("messages") or [])
        result_by_call = {
            str(message.tool_call_id): message
            for message in messages
            if isinstance(message, ToolMessage) and getattr(message, "tool_call_id", None)
        }
        call_ids = {
            str(call.get("id"))
            for message in messages
            if isinstance(message, AIMessage)
            for call in (message.tool_calls or [])
            if isinstance(call, dict) and call.get("id")
        }
        updates: list[ToolMessage] = []
        replaced_call_ids: set[str] = set()

        for transaction in store.list(RECOVERY_STATUSES):
            call_id = transaction.tool_call_id
            if transaction.status in ACTIVE_STATUSES:
                transaction = store.reconcile(transaction)
            if transaction.status == ROLLBACK_PREPARED:
                transaction = store.rollback(transaction.operation_id)
            if transaction.status == PREPARED and context.get("resume_file_writes") is True:
                transaction = store.continue_prepared(transaction)
            existing_result = result_by_call.get(call_id)
            if existing_result is not None:
                if transaction.status == APPLIED and existing_result.status != "error":
                    store.commit(transaction.operation_id)
                    continue
                if transaction.status not in {CONFLICT, ROLLED_BACK, ABANDONED}:
                    continue
                replaced_call_ids.add(call_id)
            if call_id not in call_ids:
                # A transaction without its originating model call cannot be
                # safely associated with this thread history; leave it visible
                # to /writes for explicit user action.
                continue
            if transaction.status == APPLIED:
                message = ToolMessage(
                    content=f"OK (recovered file write {transaction.operation_id})",
                    name=transaction.tool_name,
                    tool_call_id=call_id,
                    additional_kwargs={"my_assistant_file_transaction": transaction.operation_id},
                )
            elif transaction.status == CONFLICT:
                message = ToolMessage(
                    content=f"Error: file write recovery stopped for {transaction.display_path}: {transaction.error or 'file changed outside the transaction'}",
                    name=transaction.tool_name,
                    tool_call_id=call_id,
                    status="error",
                    id=getattr(existing_result, "id", None),
                    additional_kwargs={"my_assistant_file_transaction": transaction.operation_id},
                )
            elif transaction.status in {ROLLED_BACK, ABANDONED}:
                disposition = "rolled back" if transaction.status == ROLLED_BACK else "abandoned"
                message = ToolMessage(
                    content=f"Error: file write {transaction.operation_id} was {disposition} by the user during recovery.",
                    name=transaction.tool_name,
                    tool_call_id=call_id,
                    status="error",
                    id=getattr(existing_result, "id", None),
                    additional_kwargs={"my_assistant_file_transaction": transaction.operation_id},
                )
            else:
                continue
            updates.append(message)
            result_by_call[call_id] = message

        if not updates:
            return None

        # Insert the recovered tool result immediately after its AI tool call,
        # before any new user message that started the recovery turn.
        pending = {str(message.tool_call_id): message for message in updates}
        rebuilt = []
        for message in messages:
            if isinstance(message, ToolMessage) and str(message.tool_call_id) in replaced_call_ids:
                continue
            rebuilt.append(message)
            if isinstance(message, AIMessage):
                for call in message.tool_calls or []:
                    call_id = str(call.get("id") or "") if isinstance(call, dict) else ""
                    recovered = pending.pop(call_id, None)
                    if recovered is not None:
                        rebuilt.append(recovered)
        if pending:
            rebuilt.extend(pending.values())
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *rebuilt]}
