from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

from my_assistant.agents.middlewares import (
    file_transaction_recovery_middleware as recovery_module,
)
from my_assistant.agents.middlewares.file_transaction_recovery_middleware import (
    FileTransactionRecoveryMiddleware,
)
from my_assistant.sandbox.file_transactions import (
    APPLIED,
    COMMITTED,
    CONFLICT,
    PREPARED,
    ROLLBACK_PREPARED,
    ROLLED_BACK,
    FileTransactionConflict,
    FileTransactionStore,
    hash_path,
    staged_content_path,
)


def _prepare(store: FileTransactionStore, target: Path, *, before: bytes | None, after: bytes):
    operation_id = "tool-call-1"
    staged = staged_content_path(target, operation_id)
    return store.prepare(
        operation_id=operation_id,
        tool_call_id=operation_id,
        tool_name="write_file",
        display_path=str(target),
        target_path=str(target),
        pre_content=before,
        expected_content=after,
        staged_path=str(staged),
    )


def test_prepared_row_contains_recoverable_content_before_stage_exists(tmp_path: Path):
    target = tmp_path / "note.txt"
    target.write_bytes(b"before")
    store = FileTransactionStore(tmp_path / "runtime", "thread-1")

    transaction = _prepare(store, target, before=b"before", after=b"after")

    assert transaction.status == PREPARED
    assert transaction.expected_content == b"after"
    assert not Path(transaction.staged_path).exists()
    assert store.continue_prepared(transaction).status == APPLIED
    assert target.read_bytes() == b"after"


def test_reconcile_expected_hash_recovers_replace_before_applied_log(tmp_path: Path):
    target = tmp_path / "note.txt"
    target.write_bytes(b"after")
    store = FileTransactionStore(tmp_path / "runtime", "thread-1")
    transaction = _prepare(store, target, before=b"before", after=b"after")

    recovered = store.reconcile(transaction)

    assert recovered.status == APPLIED
    assert hash_path(target) == recovered.expected_hash
    assert store.commit(recovered.operation_id).status == COMMITTED


def test_reconcile_external_change_conflicts_and_rollback_refuses(tmp_path: Path):
    target = tmp_path / "note.txt"
    target.write_bytes(b"before")
    store = FileTransactionStore(tmp_path / "runtime", "thread-1")
    transaction = _prepare(store, target, before=b"before", after=b"after")
    target.write_bytes(b"external")

    conflicted = store.reconcile(transaction)

    assert conflicted.status == CONFLICT
    with pytest.raises(FileTransactionConflict):
        store.rollback(transaction.operation_id)


def test_rollback_restores_snapshot_and_new_file_removal(tmp_path: Path):
    target = tmp_path / "note.txt"
    target.write_bytes(b"before")
    store = FileTransactionStore(tmp_path / "runtime", "thread-1")
    transaction = _prepare(store, target, before=b"before", after=b"after")
    store.continue_prepared(transaction)

    rolled_back = store.rollback(transaction.operation_id)

    assert rolled_back.status == ROLLED_BACK
    assert target.read_bytes() == b"before"


def test_undo_latest_write_restores_the_most_recent_committed_operation(tmp_path: Path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"old-1")
    second.write_bytes(b"old-2")
    store = FileTransactionStore(tmp_path / "runtime", "thread-1")

    first_tx = _prepare(store, first, before=b"old-1", after=b"new-1")
    store.continue_prepared(first_tx)
    store.commit(first_tx.operation_id)

    second_tx = store.prepare(
        operation_id="tool-call-2",
        tool_call_id="tool-call-2",
        tool_name="write_file",
        display_path=str(second),
        target_path=str(second),
        pre_content=b"old-2",
        expected_content=b"new-2",
        staged_path=str(staged_content_path(second, "tool-call-2")),
    )
    store.continue_prepared(second_tx)
    store.commit(second_tx.operation_id)

    latest = max(store.list((COMMITTED,)), key=lambda item: (item.created_at, item.operation_id))
    rolled_back = store.rollback(latest.operation_id)

    assert rolled_back.status == ROLLED_BACK
    assert second.read_bytes() == b"old-2"
    assert first.read_bytes() == b"new-1"


def test_rollback_refuses_latest_write_after_external_change(tmp_path: Path):
    target = tmp_path / "note.txt"
    target.write_bytes(b"before")
    store = FileTransactionStore(tmp_path / "runtime", "thread-1")
    transaction = _prepare(store, target, before=b"before", after=b"after")
    store.continue_prepared(transaction)
    store.commit(transaction.operation_id)
    target.write_bytes(b"external")

    with pytest.raises(FileTransactionConflict):
        store.rollback(transaction.operation_id)


def test_rollback_intent_recovers_both_crash_windows(tmp_path: Path):
    target = tmp_path / "note.txt"
    target.write_bytes(b"before")
    store = FileTransactionStore(tmp_path / "runtime", "thread-1")
    transaction = _prepare(store, target, before=b"before", after=b"after")
    store.continue_prepared(transaction)

    pending = store.transition(transaction.operation_id, ROLLBACK_PREPARED)
    assert store.rollback(pending.operation_id).status == ROLLED_BACK
    assert target.read_bytes() == b"before"

    target.write_bytes(b"after")
    store.transition(transaction.operation_id, ROLLBACK_PREPARED)
    target.write_bytes(b"before")  # replacement completed, final state write did not
    assert store.reconcile(store.get(transaction.operation_id)).status == ROLLED_BACK


def test_operation_id_is_idempotent_but_cannot_be_reused_for_other_content(tmp_path: Path):
    target = tmp_path / "note.txt"
    target.write_bytes(b"before")
    store = FileTransactionStore(tmp_path / "runtime", "thread-1")
    transaction = _prepare(store, target, before=b"before", after=b"after")

    assert _prepare(store, target, before=b"before", after=b"after") == transaction
    with pytest.raises(FileTransactionConflict):
        _prepare(store, target, before=b"before", after=b"different")


def test_recovery_middleware_continues_and_inserts_missing_tool_message(tmp_path: Path, monkeypatch):
    target = tmp_path / "note.txt"
    target.write_bytes(b"before")
    store = FileTransactionStore(tmp_path / "runtime", "thread-1")
    transaction = _prepare(store, target, before=b"before", after=b"after")
    monkeypatch.setattr(recovery_module, "transaction_root", lambda *args, **kwargs: tmp_path / "runtime")
    middleware = FileTransactionRecoveryMiddleware()
    state = {"messages": [AIMessage(content="", tool_calls=[{"name": "write_file", "args": {}, "id": "tool-call-1", "type": "tool_call"}])]}

    update = middleware.before_agent(state, Runtime(context={"thread_id": "thread-1", "resume_file_writes": True}))

    assert update is not None
    recovered = next(message for message in update["messages"] if isinstance(message, ToolMessage))
    assert recovered.tool_call_id == "tool-call-1"
    assert recovered.status != "error"
    assert target.read_bytes() == b"after"
    assert store.get(transaction.operation_id).status == APPLIED


def test_recovery_middleware_replaces_success_result_after_rollback(tmp_path: Path, monkeypatch):
    target = tmp_path / "note.txt"
    target.write_bytes(b"before")
    store = FileTransactionStore(tmp_path / "runtime", "thread-1")
    transaction = _prepare(store, target, before=b"before", after=b"after")
    store.continue_prepared(transaction)
    store.rollback(transaction.operation_id)
    monkeypatch.setattr(recovery_module, "transaction_root", lambda *args, **kwargs: tmp_path / "runtime")
    middleware = FileTransactionRecoveryMiddleware()
    old_result = ToolMessage(content="OK", tool_call_id="tool-call-1", name="write_file")
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[{"name": "write_file", "args": {}, "id": "tool-call-1", "type": "tool_call"}]),
            old_result,
        ]
    }

    update = middleware.before_agent(state, Runtime(context={"thread_id": "thread-1"}))

    assert update is not None
    results = [message for message in update["messages"] if isinstance(message, ToolMessage)]
    assert len(results) == 1
    assert results[0].status == "error"


def test_recovery_middleware_finishes_pending_rollback_before_reading_old_result(tmp_path: Path, monkeypatch):
    target = tmp_path / "note.txt"
    target.write_bytes(b"before")
    store = FileTransactionStore(tmp_path / "runtime", "thread-1")
    transaction = _prepare(store, target, before=b"before", after=b"after")
    store.continue_prepared(transaction)
    store.transition(transaction.operation_id, ROLLBACK_PREPARED)
    monkeypatch.setattr(recovery_module, "transaction_root", lambda *args, **kwargs: tmp_path / "runtime")
    middleware = FileTransactionRecoveryMiddleware()
    state = {
        "messages": [
            AIMessage(content="", tool_calls=[{"name": "write_file", "args": {}, "id": "tool-call-1", "type": "tool_call"}]),
            ToolMessage(content="OK", tool_call_id="tool-call-1", name="write_file"),
        ]
    }

    update = middleware.before_agent(state, Runtime(context={"thread_id": "thread-1"}))

    assert update is not None
    assert target.read_bytes() == b"before"
    assert store.get(transaction.operation_id).status == ROLLED_BACK
    results = [message for message in update["messages"] if isinstance(message, ToolMessage)]
    assert len(results) == 1
    assert results[0].status == "error"
