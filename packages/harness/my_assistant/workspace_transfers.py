"""Explicit user-operated snapshots; never mount another Workspace live."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import uuid
from contextlib import closing

from my_assistant.config.workspace import Workspace
from my_assistant.utils.thread_id import validate_thread_id
from my_assistant.utils.time import now_iso

MAX_EXPORT_BYTES = 1024 * 1024


def _workspace(registry, context):
    workspace = Workspace.from_context(context)
    if workspace is None:
        raise PermissionError("Workspace is unavailable")
    return registry.validate(workspace)


def _schema(db):
    db.execute("CREATE TABLE IF NOT EXISTS workspace_exports (id TEXT PRIMARY KEY, manifest TEXT NOT NULL, content BLOB NOT NULL)")


def list_exports(registry):
    """Return export metadata without disclosing stored snapshot bytes."""
    with closing(registry._connect()) as db, db:
        _schema(db)
        rows = db.execute("SELECT manifest FROM workspace_exports ORDER BY rowid DESC").fetchall()
    return [json.loads(row[0]) for row in rows]


def forget_export(registry, export_id):
    """Revoke future references; already attached thread copies remain local."""
    if not re.fullmatch(r"export-[0-9a-f]{32}", export_id):
        raise ValueError("Invalid export ID")
    with closing(registry._connect()) as db, db:
        _schema(db)
        cursor = db.execute("DELETE FROM workspace_exports WHERE id=?", (export_id,))
        if cursor.rowcount != 1:
            raise LookupError("Export not found")


def export_file(registry, context, path):
    workspace = _workspace(registry, context)
    target = workspace.check(path)
    before = target.lstat()
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(target, flags)
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise PermissionError("Export target identity changed")
        workspace.check(target)
        if opened.st_size > MAX_EXPORT_BYTES:
            raise ValueError("Export is limited to 1 MiB of UTF-8 text")
        content = stream.read(MAX_EXPORT_BYTES + 1)
        after = os.fstat(stream.fileno())
    if len(content) > MAX_EXPORT_BYTES or (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise PermissionError("File changed while exporting")
    workspace.check(target)
    current = target.stat()
    if (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise PermissionError("File changed while exporting")
    text = content.decode("utf-8")
    from my_assistant.agents.memory.policy import is_sensitive_memory_text

    if "\0" in text or is_sensitive_memory_text(text):
        raise PermissionError("Sensitive or binary content cannot be exported")
    manifest = {"id": f"export-{uuid.uuid4().hex}", "workspace_id": workspace.id, "root_fingerprint": workspace.root_fingerprint, "relative_path": target.relative_to(workspace.root).as_posix(), "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content), "created_at": now_iso()}
    with closing(registry._connect()) as db, db:
        _schema(db)
        db.execute("INSERT INTO workspace_exports VALUES (?,?,?)", (manifest["id"], json.dumps(manifest), content))
    return manifest


def reference_export(registry, context, thread_id, export_id):
    validate_thread_id(thread_id)
    workspace = _workspace(registry, context)
    if not re.fullmatch(r"export-[0-9a-f]{32}", export_id):
        raise ValueError("Invalid export ID")
    registry.check(thread_id, {}, workspace)
    with closing(registry._connect()) as db, db:
        _schema(db)
        row = db.execute("SELECT manifest,content FROM workspace_exports WHERE id=?", (export_id,)).fetchone()
    if row is None:
        raise LookupError("Export not found in this user's catalog")
    manifest = json.loads(row[0])
    content = bytes(row[1])
    if len(content) > MAX_EXPORT_BYTES or hashlib.sha256(content).hexdigest() != manifest["sha256"]:
        raise PermissionError("Export content failed integrity validation")
    # Exported snapshots survive source unregistration. This explicit reference
    # grants only these bytes, no access to the live source or its index/task.
    relative = f".my-assistant/threads/{thread_id}/uploads/{export_id}.txt"
    target = workspace.check(relative)
    workspace.resolve(target.parent).mkdir(parents=True, exist_ok=True)
    workspace.check(target)
    # Publish a fully flushed snapshot without overwriting an existing target.
    # A crash before publication leaves at most an unreferenced temporary file;
    # it cannot expose a partial attachment at the stable virtual path.
    fd, temporary = tempfile.mkstemp(prefix=f".{export_id}-", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        registry.validate(workspace)
        workspace.check(target)
        workspace.resolve(temporary)
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            workspace.check(target)
            if target.stat().st_size > MAX_EXPORT_BYTES or hashlib.sha256(target.read_bytes()).hexdigest() != manifest["sha256"]:
                raise PermissionError("Reference snapshot was modified; refusing overwrite") from None
    finally:
        os.unlink(temporary)
    return {**manifest, "target_workspace_id": workspace.id, "virtual_path": f"/mnt/user-data/uploads/{export_id}.txt", "snapshot": True}
