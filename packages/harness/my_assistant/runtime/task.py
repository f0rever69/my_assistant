"""Thread-checkpoint persistence for structured execution tasks."""

from __future__ import annotations

import copy
from contextlib import AbstractAsyncContextManager
from typing import Any

from langgraph.checkpoint.base import uuid6

from my_assistant.agents.task_continuity.task_state import (
    TaskState,
    normalize_task_state,
)
from my_assistant.runtime.goal import (
    _call_checkpointer_method,
    _checkpoint_id_from_tuple,
    _next_channel_version,
    ensure_thread_checkpoint,
)
from my_assistant.runtime.keyed_lock import AsyncKeyedLockTable
from my_assistant.utils.time import now_iso

_task_locks = AsyncKeyedLockTable[str]()


def task_thread_lock(thread_id: str) -> AbstractAsyncContextManager[None]:
    return _task_locks.hold(thread_id)


async def read_thread_task(checkpointer: Any, thread_id: str) -> TaskState | None:
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoint_tuple = await _call_checkpointer_method(
        checkpointer, "aget_tuple", "get_tuple", config
    )
    if checkpoint_tuple is None:
        return None
    checkpoint = getattr(checkpoint_tuple, "checkpoint", {}) or {}
    values = (
        checkpoint.get("channel_values", {}) if isinstance(checkpoint, dict) else {}
    )
    return (
        normalize_task_state(values.get("task")) if isinstance(values, dict) else None
    )


async def write_thread_task(
    checkpointer: Any,
    thread_id: str,
    task: TaskState | None,
    *,
    create_if_missing: bool = False,
) -> TaskState | None:
    if create_if_missing:
        await ensure_thread_checkpoint(checkpointer, thread_id)
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    checkpoint_tuple = await _call_checkpointer_method(
        checkpointer, "aget_tuple", "get_tuple", config
    )
    if checkpoint_tuple is None:
        raise LookupError(f"Thread {thread_id} checkpoint not found")
    checkpoint = dict(getattr(checkpoint_tuple, "checkpoint", {}) or {})
    metadata = dict(getattr(checkpoint_tuple, "metadata", {}) or {})
    values = dict(checkpoint.get("channel_values", {}) or {})
    normalized = normalize_task_state(task) if task is not None else None
    if normalized is None:
        values.pop("task", None)
    else:
        values["task"] = copy.deepcopy(normalized)
    versions = dict(checkpoint.get("channel_versions", {}) or {})
    next_version = _next_channel_version(checkpointer, versions.get("task"))
    versions["task"] = next_version
    changed_versions = {"task": next_version}
    binding = (normalized or {}).get("workspace_binding")
    if binding is not None:
        from my_assistant.workspaces import state_binding

        state_binding({**values, "task": normalized})
        values["workspace_binding"] = copy.deepcopy(binding)
        binding_version = _next_channel_version(checkpointer, versions.get("workspace_binding"))
        versions["workspace_binding"] = binding_version
        changed_versions["workspace_binding"] = binding_version
    checkpoint["channel_values"] = values
    checkpoint["channel_versions"] = versions
    checkpoint["id"] = str(uuid6())
    metadata.update(
        updated_at=now_iso(),
        source="update",
        step=metadata.get("step", 0) + 1,
        writes={"task": {"task": normalized}},
    )
    write_config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": _checkpoint_id_from_tuple(checkpoint_tuple),
        }
    }
    await _call_checkpointer_method(
        checkpointer,
        "aput",
        "put",
        write_config,
        checkpoint,
        metadata,
        changed_versions,
    )
    return normalized
