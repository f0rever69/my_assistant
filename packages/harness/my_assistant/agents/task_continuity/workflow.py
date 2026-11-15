"""Versioned task contract and execution policy; todos remain the only step list.

No model report is an acceptance verdict. Unknown semantic criteria require a
human decision. All file evidence is read through the current Workspace policy.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
from pathlib import Path

from my_assistant.config.workspace import Workspace
from my_assistant.utils.time import now_iso

MAX_CRITERIA = 12
MAX_STEPS = 30
MAX_ACTIONS = 30
PHASES = {"preview", "awaiting_approval", "executing", "delivered", "completed", "blocked", "paused", "abandoned"}
READ_TOOLS = frozenset({"read_file", "ls", "glob", "grep", "workspace_search", "history_search", "history_read", "view_image", "list_uploaded_files"})
CONTROL_TOOLS = frozenset({"plan_submit", "write_todos", "task_update", "ask_clarification"})
TRUSTED_TOOL_MODULES = {
    **dict.fromkeys({"read_file", "ls", "glob", "grep"}, "my_assistant.sandbox.tools"),
    "workspace_search": "my_assistant.sandbox.workspace_search",
    **dict.fromkeys({"history_search", "history_read", "task_update"}, "my_assistant.agents.task_continuity.tools"),
    "plan_submit": "my_assistant.agents.middlewares.task_workflow_middleware",
    "write_todos": "langchain.agents.middleware.todo",
    "view_image": "my_assistant.tools.builtins.view_image_tool",
    "list_uploaded_files": "my_assistant.tools.builtins.list_uploaded_files_tool",
    "ask_clarification": "my_assistant.tools.builtins.clarification_tool",
}


def wants_preview(text: str) -> bool:
    return bool(re.search(r"(?:先.{0,8}(?:规划|计划).{0,40}(?:确认|批准).{0,15}(?:执行|动手)|(?:plan|planning).{0,60}(?:wait|approval|confirm).{0,30}(?:execut|proceed)?)", text, re.IGNORECASE | re.DOTALL))


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalize_todos(value: object) -> list[dict]:
    if not isinstance(value, (list, tuple)):
        return []
    return [{"content": row["content"].strip()[:1000], "status": row["status"]}
            for row in value[:MAX_STEPS] if isinstance(row, dict) and isinstance(row.get("content"), str)
            and row["content"].strip() and row.get("status") in {"pending", "in_progress", "completed"}]


def plan_digest(task: dict, todos: object) -> str:
    execution = task["execution"]
    return digest({"objective": task["objective"], "steps": [row["content"] for row in normalize_todos(todos)],
                   "actions": execution["allowed_actions"], "criteria": task.get("acceptance_criteria", [])})


def event(task: dict, kind: str, *, todos: object = None, **details) -> dict:
    task = copy.deepcopy(task)
    execution = task["execution"]
    if todos is not None:
        # Recovery notes are a bounded projection of Todo, not another plan.
        rows = normalize_todos(todos)
        task["completed_steps"] = [row["content"] for row in rows if row["status"] == "completed"]
        task["pending_steps"] = [row["content"] for row in rows if row["status"] != "completed"]
    execution["revision"] += 1
    execution["events"] = [*execution.get("events", []), {"version": 1, "revision": execution["revision"],
        "plan_version": execution["plan_version"], "task_revision": task.get("revision", 1) + 1,
        "type": kind, "at": now_iso(), **details}][-40:]
    task["revision"] = task.get("revision", 1) + 1
    task["updated_at"] = now_iso()
    return task


def begin_preview(task: dict) -> dict:
    task = copy.deepcopy(task)
    old = task.get("execution") or {}
    task["execution"] = {"schema_version": 1, "phase": "preview", "require_approval": True,
        "plan_version": old.get("plan_version", 0), "revision": old.get("revision", 0),
        "allowed_actions": [], "plan_digest": "", "approval": None, "checks": [],
        "blocker": None, "events": old.get("events", [])}
    task["status"] = "active"
    return event(task, "preview_started")


def normalize_actions(actions: object) -> list[dict]:
    if not isinstance(actions, list) or len(actions) > MAX_ACTIONS:
        raise ValueError("At most 30 allowed actions are supported")
    result = []
    for action in actions:
        if not isinstance(action, dict) or set(action) != {"tool", "args"} or not isinstance(action["tool"], str) or not isinstance(action["args"], dict):
            raise ValueError("Each allowed action needs tool and args")
        name, args = action["tool"], action["args"]
        if not name or len(name) > 100 or len(json.dumps(args)) > 8000:
            raise ValueError("Invalid action or oversized arguments")
        # File contents may be authored later, but the exact target is approved.
        # Shell and external calls require their complete arguments to match.
        if name in {"write_file", "str_replace", "delete_file"} and (set(args) != {"path"} or not isinstance(args["path"], str) or not args["path"]):
            raise ValueError("File action grants must contain only an exact path")
        if name in {"task", "batch_task", "invoke_acp_agent", "skill_manage", "setup_agent", "update_agent"}:
            raise ValueError("Delegation and agent self-modification are unavailable in approval-controlled tasks")
        result.append(copy.deepcopy(action))
    return result


def criteria_path(workspace: Workspace | None, path: str, thread_id: str) -> Path:
    if workspace is None:
        raise PermissionError("File acceptance requires a project Workspace")
    if path.startswith("/mnt/user-data/workspace/"):
        path = path.removeprefix("/mnt/user-data/workspace/")
    elif path.startswith("/mnt/user-data/outputs/"):
        path = str(workspace.runtime_dir(thread_id) / "outputs" / path.removeprefix("/mnt/user-data/outputs/"))
    return workspace.check(path)


def file_hash(path: Path) -> str:
    # Bounded work, streaming bytes, no Shell and no whole-file model context.
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > 64 * 1024 * 1024:
        raise ValueError("Expected a regular file of at most 64 MiB")
    checksum = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise PermissionError("File identity changed during acceptance")
        remaining = 64 * 1024 * 1024 + 1
        while remaining:
            block = stream.read(min(64 * 1024, remaining))
            if not block:
                break
            checksum.update(block)
            remaining -= len(block)
        after = os.fstat(stream.fileno())
    current = path.lstat()
    if remaining == 0 or (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise PermissionError("File changed during acceptance")
    return checksum.hexdigest()


def prepare_criteria(criteria: object, workspace: Workspace | None, thread_id: str) -> list[dict]:
    if not isinstance(criteria, list) or not criteria or len(criteria) > MAX_CRITERIA:
        raise ValueError("A task requires 1-12 acceptance criteria")
    result = []
    for index, raw in enumerate(criteria):
        if not isinstance(raw, dict) or raw.get("kind") not in {"file_exists", "file_nonempty", "file_unchanged", "human"}:
            raise ValueError("Criterion kind must be file_exists, file_nonempty, file_unchanged or human")
        description = raw.get("description", "")
        if not isinstance(description, str) or not description.strip() or len(description) > 1000:
            raise ValueError("Each criterion needs a bounded description")
        row = {"id": f"c{index + 1}", "kind": raw["kind"], "description": description.strip()}
        if raw["kind"] != "human":
            if not isinstance(raw.get("path"), str) or not raw["path"]:
                raise ValueError("File criteria require an exact path")
            row["path"] = raw["path"]
            path = criteria_path(workspace, row["path"], thread_id)
            if raw["kind"] == "file_unchanged":
                row["pre_hash"] = file_hash(path)
                criteria_path(workspace, row["path"], thread_id)
        result.append(row)
    return result


def submit(task: dict, steps: list[str], actions: list[dict], criteria: list[dict]) -> tuple[dict, list[dict]]:
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_STEPS or any(not isinstance(s, str) or not s.strip() or len(s) > 1000 for s in steps):
        raise ValueError("A plan requires 1-30 bounded steps")
    task = copy.deepcopy(task)
    execution = task.get("execution")
    if execution is None:
        task = begin_preview(task)
        execution = task["execution"]
        execution["require_approval"] = False
    # Baselines are captured once. Replanning cannot bless an already changed source.
    old = {(c.get("kind"), c.get("path")): c for c in task.get("acceptance_criteria", [])}
    for row in criteria:
        if row["kind"] == "file_unchanged" and (row["kind"], row["path"]) in old:
            row["pre_hash"] = old[(row["kind"], row["path"])]["pre_hash"]
    task["acceptance_criteria"] = criteria
    execution["allowed_actions"] = normalize_actions(actions)
    todos = [{"content": step.strip(), "status": "pending"} for step in steps]
    new_digest = plan_digest(task, todos)
    if new_digest != execution.get("plan_digest"):
        execution["plan_version"] += 1
        execution["approval"] = None
    execution.update(plan_digest=new_digest, checks=[], blocker=None,
        phase="awaiting_approval" if execution["require_approval"] and execution["approval"] is None else "executing")
    task["status"] = "active"
    return event(task, "plan_submitted", todos=todos), todos


def approve(task: dict, todos: list, version: int) -> dict:
    task = copy.deepcopy(task)
    execution = task["execution"]
    if execution["phase"] != "awaiting_approval" or version != execution["plan_version"] or execution["plan_digest"] != plan_digest(task, todos):
        raise ValueError("Plan changed or version is stale; review the current plan")
    execution["approval"] = {"plan_version": version, "plan_digest": execution["plan_digest"],
        "actions_digest": digest(execution["allowed_actions"]), "at": now_iso(), "source": "user"}
    execution["phase"] = "executing"
    return event(task, "plan_approved")


def tool_allowed(task: dict | None, todos: list, name: str, args: dict) -> bool:
    execution = (task or {}).get("execution")
    if not execution:
        return True
    if execution["phase"] in {"paused", "abandoned"}:
        return name in READ_TOOLS
    if execution["phase"] == "completed":
        return name in READ_TOOLS or name in {"plan_submit", "ask_clarification"}
    if execution["phase"] in {"blocked", "delivered"}:
        return name in READ_TOOLS or name in {"plan_submit", "ask_clarification", "task_update"}
    if name in READ_TOOLS or name in CONTROL_TOOLS:
        return True
    if not execution["require_approval"]:
        return execution["phase"] not in {"blocked", "delivered", "completed", "paused", "abandoned"}
    approval = execution.get("approval") or {}
    if execution["phase"] != "executing" or approval.get("source") != "user" or approval.get("plan_version") != execution["plan_version"] or approval.get("plan_digest") != plan_digest(task, todos) or approval.get("actions_digest") != digest(execution["allowed_actions"]):
        return False
    projected = {"path": args.get("path")} if name in {"write_file", "str_replace", "delete_file"} else args
    return {"tool": name, "args": projected} in execution["allowed_actions"]


def resume(task: dict, todos: list) -> dict:
    """Trusted continuation; a blocked preview cannot bypass its missing approval."""
    task = copy.deepcopy(task)
    execution = task["execution"]
    if execution["phase"] in {"preview", "awaiting_approval", "completed", "abandoned"}:
        raise ValueError("Review/approve the plan before continuing; accepted or abandoned tasks cannot resume")
    if execution["require_approval"]:
        approval = execution.get("approval") or {}
        valid = (approval.get("source") == "user"
                 and approval.get("plan_version") == execution["plan_version"]
                 and approval.get("plan_digest") == plan_digest(task, todos)
                 and approval.get("actions_digest") == digest(execution["allowed_actions"]))
        if not valid:
            raise ValueError("Approval missing or stale; use /plan preview to submit a reviewable plan")
    execution.update(phase="executing", blocker=None, checks=[])
    task["status"] = "active"
    return event(task, "user_requested_continue", todos=todos)


def block(task: dict, reason: str, options: list[str]) -> dict:
    task = copy.deepcopy(task)
    task["execution"].update(phase="blocked", blocker={"reason": reason[:1000], "options": options[:5]})
    task["status"] = "blocked"
    task["next_action"] = "; ".join(options)[:2000]
    return event(task, "task_blocked")


def check_delivery(task: dict, workspace: Workspace | None, thread_id: str, *, human_confirm: bool = False, todos: object = None) -> dict:
    task = copy.deepcopy(task)
    execution = task["execution"]
    checks = []
    for row in task.get("acceptance_criteria", []):
        status, detail = "unverified", "需要人工验收；模型自评不能替代验收"
        if row["kind"] == "human":
            if human_confirm:
                status, detail = "passed", "用户明确确认；仅适用于本次交付"
        else:
            try:
                path = criteria_path(workspace, row["path"], thread_id)
                if row["kind"] == "file_unchanged":
                    holds = file_hash(path) == row["pre_hash"]
                else:
                    holds = path.is_file() and (row["kind"] == "file_exists" or path.stat().st_size > 0)
                criteria_path(workspace, row["path"], thread_id)
                status, detail = ("passed", "代码校验通过") if holds else ("failed", "实际文件不符合条件")
            except (OSError, ValueError) as exc:
                status, detail = "failed", str(exc)[:200]
        checks.append({"criterion_id": row["id"], "status": status, "detail": detail})
    # Human acceptance is a final action, and deterministic checks rerun at that boundary.
    steps_complete = todos is None or all(row["status"] == "completed" for row in normalize_todos(todos))
    accepted = bool(checks) and all(c["status"] == "passed" for c in checks) and human_confirm and steps_complete
    execution.update(checks=checks, phase="completed" if accepted else "delivered", blocker=None)
    task["status"] = "completed" if accepted else "delivered"
    task["next_action"] = "" if accepted else "Review acceptance results; repair failed checks, then /accept confirm."
    return event(task, "acceptance_confirmed" if accepted else "delivery_checked", todos=todos)
