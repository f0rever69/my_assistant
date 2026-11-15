"""Bounded, checkpointed execution state for resumable user tasks."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Literal, NotRequired, TypedDict

from langgraph.channels import BinaryOperatorAggregate

from my_assistant.utils.time import now_iso

TaskStatus = Literal["active", "paused", "blocked", "delivered", "completed", "abandoned"]
TASK_STATUSES = frozenset({"active", "paused", "blocked", "delivered", "completed", "abandoned"})
MAX_TASK_TEXT = 2000
MAX_TASK_ITEMS = 50
MAX_TASK_ITEM_CHARS = 1000


class TaskState(TypedDict):
    workspace_binding: NotRequired[dict]
    acceptance_criteria: NotRequired[list[dict]]
    execution: NotRequired[dict]
    objective: str
    status: TaskStatus
    completed_steps: list[str]
    pending_steps: list[str]
    decisions: list[str]
    constraints: list[str]
    modified_files: list[str]
    failed_attempts: list[str]
    next_action: str
    created_at: str
    updated_at: str
    revision: int


def _text(value: object, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _items(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = _text(item, MAX_TASK_ITEM_CHARS)
        if text and text not in result:
            result.append(text)
    return result[-MAX_TASK_ITEMS:]


def normalize_task_state(value: object) -> TaskState | None:
    if not isinstance(value, dict):
        return None
    objective = _text(value.get("objective"), MAX_TASK_TEXT)
    status = value.get("status")
    if not objective or status not in TASK_STATUSES:
        return None
    revision = value.get("revision", 1)
    if type(revision) is not int or revision < 1:
        revision = 1
    created_at = _text(value.get("created_at"), 64) or now_iso()
    updated_at = _text(value.get("updated_at"), 64) or created_at
    result = {
        "objective": objective,
        "status": status,
        "completed_steps": _items(value.get("completed_steps")),
        "pending_steps": _items(value.get("pending_steps")),
        "decisions": _items(value.get("decisions")),
        "constraints": _items(value.get("constraints")),
        "modified_files": _items(value.get("modified_files")),
        "failed_attempts": _items(value.get("failed_attempts")),
        "next_action": _text(value.get("next_action"), MAX_TASK_TEXT),
        "created_at": created_at,
        "updated_at": updated_at,
        "revision": revision,
    }
    if isinstance(value.get("workspace_binding"), dict):
        result["workspace_binding"] = dict(value["workspace_binding"])
    if "execution" in value:
        from my_assistant.agents.task_continuity.workflow import PHASES

        execution = value["execution"]
        if not isinstance(execution, dict) or execution.get("phase") not in PHASES or type(execution.get("require_approval")) is not bool:
            raise ValueError("Invalid task execution policy; refusing to drop its authorization gate")
        result["execution"] = copy.deepcopy(execution)
        criteria = value.get("acceptance_criteria", [])
        if not isinstance(criteria, list) or len(criteria) > 12 or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("description"), str) or row.get("kind") not in {"file_exists", "file_nonempty", "file_unchanged", "human"} for row in criteria):
            raise ValueError("Invalid task acceptance criteria; refusing to lose verification state")
        result["acceptance_criteria"] = copy.deepcopy(criteria)
    return result


def create_task_state(objective: str) -> TaskState:
    normalized = _text(objective, MAX_TASK_TEXT)
    if not normalized:
        raise ValueError("Task objective must not be empty")
    timestamp = now_iso()
    return {
        "objective": normalized,
        "status": "active",
        "completed_steps": [],
        "pending_steps": [],
        "decisions": [],
        "constraints": [],
        "modified_files": [],
        "failed_attempts": [],
        "next_action": "",
        "created_at": timestamp,
        "updated_at": timestamp,
        "revision": 1,
    }


def update_task_progress(value: object, patch: dict) -> TaskState:
    task = normalize_task_state(value)
    if task is None:
        raise ValueError("No valid task is active")
    next_task = copy.deepcopy(task)
    list_fields = {
        "completed_steps": "add_completed_steps",
        "pending_steps": "add_pending_steps",
        "decisions": "add_decisions",
        "constraints": "add_constraints",
        "modified_files": "add_modified_files",
        "failed_attempts": "add_failed_attempts",
    }
    for field, patch_key in list_fields.items():
        if "execution" in task and field in {"completed_steps", "pending_steps"}:
            continue  # These recovery fields are projected from versioned Todo events.
        existing = list(next_task[field])
        additions = _items(patch.get(patch_key))
        if field == "pending_steps":
            removals = set(_items(patch.get("remove_pending_steps")))
            existing = [item for item in existing if item not in removals]
        next_task[field] = _items(existing + additions)
    if isinstance(patch.get("next_action"), str):
        next_task["next_action"] = _text(patch["next_action"], MAX_TASK_TEXT)
    status = patch.get("status")
    if status is not None:
        if status not in {"blocked", "completed"}:
            raise ValueError("Model task status must be blocked or completed")
        if task["status"] != "active":
            raise ValueError("Only an active task may change status through the model tool")
        next_task["status"] = status
        if status == "completed":
            next_task["next_action"] = ""
    if "execution" in task:
        from my_assistant.agents.task_continuity.workflow import block, event

        # Model completion means delivered, never accepted. Code checks run at
        # the graph's after_agent boundary; only a trusted user can accept.
        if status == "completed":
            next_task["status"] = "delivered"
            next_task["execution"]["phase"] = "delivered"
            next_task["next_action"] = "Run acceptance checks and request user review."
        elif status == "blocked":
            return block(next_task, _text(patch.get("next_action"), 1000) or "Missing information or tool failure", ["补充资料", "修复工具后重试", "暂停或放弃任务"])
        return event(next_task, "task_progress_reported")
    next_task["updated_at"] = now_iso()
    next_task["revision"] = task["revision"] + 1
    return next_task  # type: ignore[return-value]


def merge_task_state(
    left: TaskState | None, right: TaskState | None
) -> TaskState | None:
    return (
        normalize_task_state(right) if right is not None else normalize_task_state(left)
    )


class TaskStateChannel(BinaryOperatorAggregate[TaskState | None]):
    def update(self, values: Sequence[TaskState | None]) -> bool:
        changed = super().update(values)
        if changed:
            self.value = normalize_task_state(self.value)
        return changed
