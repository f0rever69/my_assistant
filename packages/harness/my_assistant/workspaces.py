"""Local, per-user Workspace catalog and durable thread ownership.

Catalog deletion is a tombstone, never a filesystem delete. SQLite uniqueness
and BEGIN IMMEDIATE make first binding fail closed across local processes.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Any

from my_assistant.config.workspace import Workspace, fingerprint_root
from my_assistant.utils.thread_id import validate_thread_id
from my_assistant.utils.time import now_iso


class WorkspaceUnavailable(PermissionError):
    pass


class WorkspaceMismatch(PermissionError):
    pass


def workspace_binding(workspace: Workspace | None) -> dict:
    if workspace is None:
        return {"version": 1, "workspace_id": None}
    return {
        "version": 1,
        "workspace_id": workspace.id,
        "root_path": workspace.root_path,
        "root_fingerprint": workspace.root_fingerprint or fingerprint_root(workspace.root_path),
    }


def _validated_binding(binding: object) -> dict | None:
    if binding is None:
        return None
    if not isinstance(binding, dict) or binding.get("version") != 1 or "workspace_id" not in binding:
        raise WorkspaceMismatch("Malformed Workspace binding; automatic file actions are blocked")
    identifier = binding["workspace_id"]
    if identifier is None:
        if binding != workspace_binding(None):
            raise WorkspaceMismatch("Malformed thread-local Workspace binding")
    elif not isinstance(identifier, str) or not identifier or any(not isinstance(binding.get(key), str) or not binding[key] for key in ("root_path", "root_fingerprint")):
        raise WorkspaceMismatch("Incomplete Workspace binding; explicitly migrate the task")
    return dict(binding)


def state_binding(state: dict) -> dict | None:
    binding = state.get("workspace_binding")
    task = state.get("task")
    if task is not None and not isinstance(task, dict):
        raise WorkspaceMismatch("Malformed task state")
    task_binding = (task or {}).get("workspace_binding")
    binding, task_binding = _validated_binding(binding), _validated_binding(task_binding)
    if binding is not None and task_binding is not None and binding != task_binding:
        raise WorkspaceMismatch("Thread and task Workspace bindings disagree")
    if binding is not None or task_binding is not None:
        return binding if binding is not None else task_binding
    data = state.get("thread_data")
    if data:
        if not isinstance(data, dict):
            raise WorkspaceMismatch("Malformed thread data")
        policy = data.get("workspace_policy")
        if policy:
            # Legacy checkpoints without a fingerprint must not adopt a new
            # directory that happens to have the same spelling.
            if not isinstance(policy, dict) or not policy.get("root_fingerprint"):
                raise WorkspaceMismatch("Legacy Workspace has no root fingerprint. Explicitly migrate the task to a new thread.")
            return _validated_binding({"version": 1, "workspace_id": policy.get("id"), "root_path": policy.get("root_path"), "root_fingerprint": policy["root_fingerprint"]})
        return workspace_binding(None)
    return None


class WorkspaceRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "registry.sqlite3"
        with closing(self._connect()) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS workspaces (id TEXT PRIMARY KEY, path_key TEXT NOT NULL, policy TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)")
            db.execute("CREATE UNIQUE INDEX IF NOT EXISTS workspace_active_path ON workspaces(path_key) WHERE deleted=0")
            db.execute("CREATE TABLE IF NOT EXISTS thread_bindings (thread_id TEXT PRIMARY KEY, binding TEXT NOT NULL)")

    def _connect(self):
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return db

    def register(self, path: str | Path, *, name: str | None = None) -> Workspace:
        candidate = Workspace.from_path(path)
        key = os.path.normcase(candidate.root_path)
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM workspaces WHERE path_key=? AND deleted=0", (key,)).fetchone()
            if row:
                old = json.loads(row["policy"])
                if old["root_fingerprint"] != candidate.root_fingerprint:
                    raise WorkspaceUnavailable("Registered root was replaced. Remove the stale registration and register this directory as a new Workspace.")
                return self._live(old)
            candidate = replace(candidate, id=f"ws-{uuid.uuid4().hex}", name=self._name(name or candidate.name), registry_managed=True)
            stamp = now_iso()
            db.execute("INSERT INTO workspaces VALUES (?,?,?,0,?,?)", (candidate.id, key, json.dumps(candidate.to_context()), stamp, stamp))
        return candidate

    @staticmethod
    def _name(name: str) -> str:
        if not isinstance(name, str):
            raise TypeError("Workspace name must be text")
        name = name.strip()
        if not name or len(name) > 100 or any(ord(c) < 32 or ord(c) == 127 for c in name):
            raise ValueError("Workspace name must contain 1-100 visible characters")
        return name

    def record(self, ref: str) -> dict:
        with closing(self._connect()) as db:
            rows = db.execute("SELECT * FROM workspaces").fetchall()
        exact = [row for row in rows if row["id"] == ref]
        matches = exact or [row for row in rows if not row["deleted"] and json.loads(row["policy"])["name"] == ref]
        if len(matches) != 1:
            raise LookupError("Workspace ID/name is missing or ambiguous; use the full ID")
        row = matches[0]
        return {**json.loads(row["policy"]), "deleted": bool(row["deleted"]), "updated_at": row["updated_at"]}

    @staticmethod
    def _live(record: dict) -> Workspace:
        if record.get("deleted"):
            raise WorkspaceUnavailable("Workspace registration was deleted; conversation metadata is retained")
        workspace = Workspace.from_context(record)
        if workspace is None:
            raise WorkspaceUnavailable("Workspace root is missing or its fingerprint changed; file actions are unavailable")
        return workspace

    def resolve(self, ref: str) -> Workspace:
        return self._live(self.record(ref))

    def list(self) -> list[dict]:
        with closing(self._connect()) as db:
            ids = [row[0] for row in db.execute("SELECT id FROM workspaces ORDER BY updated_at DESC")]
        result = []
        for identifier in ids:
            record = self.record(identifier)
            try:
                self._live(record)
                record["available"] = True
            except WorkspaceUnavailable as exc:
                record.update(available=False, unavailable_reason=str(exc))
            result.append(record)
        return result

    def update(self, ref: str, **patch: Any) -> Workspace:
        allowed = {"name", "allowed_read_paths", "allowed_write_paths", "ignored_patterns", "shell_enabled", "network_enabled"}
        if not patch or not set(patch) <= allowed:
            raise ValueError("Only Workspace display name and policy fields may be edited")
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            # Resolve using the same transaction to avoid resurrecting tombstones.
            rows = db.execute("SELECT * FROM workspaces WHERE deleted=0").fetchall()
            exact = [row for row in rows if row["id"] == ref]
            matches = exact or [row for row in rows if json.loads(row["policy"])["name"] == ref]
            if len(matches) != 1:
                raise LookupError("Workspace ID/name is missing or ambiguous")
            record = json.loads(matches[0]["policy"])
            self._live(record)
            if "name" in patch:
                patch["name"] = self._name(patch["name"])
            for flag in ("shell_enabled", "network_enabled"):
                if flag in patch and type(patch[flag]) is not bool:
                    raise ValueError(f"{flag} must be a boolean")
            for key in ("allowed_read_paths", "allowed_write_paths", "ignored_patterns"):
                if key in patch:
                    values = patch[key]
                    if not isinstance(values, list) or any(not isinstance(item, str) or not item for item in values):
                        raise ValueError(f"{key} must be a list of relative paths/patterns")
                    if key != "ignored_patterns" and any(Path(item).is_absolute() or ":" in item or ".." in item.replace("\\", "/").split("/") for item in values):
                        raise ValueError("Scopes must stay inside the Workspace")
            record.update(patch)
            workspace = self._live(record)
            db.execute("UPDATE workspaces SET policy=?,updated_at=? WHERE id=?", (json.dumps(workspace.to_context()), now_iso(), workspace.id))
        return workspace

    def delete(self, ref: str) -> None:
        record = self.record(ref)
        with closing(self._connect()) as db, db:
            db.execute("UPDATE workspaces SET deleted=1,updated_at=? WHERE id=?", (now_iso(), record["id"]))

    def validate(self, workspace: Workspace) -> Workspace:
        if workspace.registry_managed:
            current = self.resolve(workspace.id)
            if workspace.to_context() != current.to_context():
                raise WorkspaceUnavailable("Workspace policy changed. Re-select it before continuing.")
        else:
            try:
                valid = fingerprint_root(workspace.root_path) == workspace.root_fingerprint
            except (OSError, ValueError):
                valid = False
            if not valid:
                raise WorkspaceUnavailable("Workspace root identity changed")
        return workspace

    def thread_binding(self, thread_id: str) -> dict | None:
        validate_thread_id(thread_id)
        with closing(self._connect()) as db:
            row = db.execute("SELECT binding FROM thread_bindings WHERE thread_id=?", (thread_id,)).fetchone()
        return _validated_binding(json.loads(row[0])) if row else None

    def check(self, thread_id: str, state: dict, workspace: Workspace | None, *, bind: bool = False) -> dict:
        validate_thread_id(thread_id)
        if workspace is not None:
            self.validate(workspace)
        requested = workspace_binding(workspace)
        saved = state_binding(state)
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT binding FROM thread_bindings WHERE thread_id=?", (thread_id,)).fetchone()
            owner = _validated_binding(json.loads(row[0])) if row else saved
            if owner is not None and saved is not None and owner != saved:
                raise WorkspaceMismatch("Checkpoint and local thread Workspace ownership disagree")
            if owner is not None and owner != requested:
                label = owner.get("workspace_id") or "thread-local storage"
                raise WorkspaceMismatch(f"Thread belongs to {label}. Switch back to its Workspace or explicitly migrate the task to a new thread; file actions are blocked.")
            if bind and row is None:
                db.execute("INSERT INTO thread_bindings VALUES (?,?)", (thread_id, json.dumps(requested)))
        return requested


def get_workspace_registry(user_id: str | None = None) -> WorkspaceRegistry:
    from my_assistant.config.paths import get_paths
    from my_assistant.runtime.user_context import get_effective_user_id

    return WorkspaceRegistry(get_paths().user_dir(user_id or get_effective_user_id()) / "workspaces")
