"""Stage searchable Workspace files for Docker and apply reviewed results on the host."""

import difflib
import hashlib
import json
import os
import sqlite3
import stat
import tempfile
import uuid
from contextlib import closing, contextmanager
from itertools import islice
from pathlib import Path

from my_assistant.config.paths import get_paths
from my_assistant.config.workspace import Workspace, _is_reparse_point
from my_assistant.sandbox.file_transactions import (
    APPLIED,
    FileTransactionConflict,
    FileTransactionStore,
    _atomic_write_bytes,
    hash_bytes,
    staged_content_path,
    transaction_root,
)

_PRIVATE_DIRS = {
    ".git",
    ".my-assistant",
    ".my_assistant",
    ".venv",
    ".ssh",
    ".aws",
    ".azure",
    "node_modules",
    "__pycache__",
    ".codex",
    ".agents",
}
_MAX_FILE_BYTES = 32 * 1024 * 1024
_MAX_COPY_BYTES = 512 * 1024 * 1024
_MAX_ENTRIES = 20_000


def workspace_copy_key(workspace: Workspace) -> str:
    policy = json.dumps(workspace.to_context(), sort_keys=True)
    return hashlib.sha256(f"docker-workspace-v1:{policy}".encode()).hexdigest()[:16]


def _copy_root(workspace: Workspace, thread_id: str, user_id: str) -> Path:
    return (
        get_paths().thread_dir(thread_id, user_id=user_id)
        / "docker-workspace"
        / workspace_copy_key(workspace)
    )


@contextmanager
def _copy_lock(root: Path):
    with closing(sqlite3.connect(root.with_suffix(".lock"), timeout=30)) as lock, lock:
        lock.execute("BEGIN IMMEDIATE")
        yield


def _read_regular(path: Path) -> bytes:
    def check(info: os.stat_result) -> None:
        if (
            not stat.S_ISREG(info.st_mode)
            or _is_reparse_point(info)
            or info.st_nlink != 1
        ):
            raise PermissionError(
                f"Workspace copy requires a regular, unlinked file: {path.name}"
            )
        if info.st_size > _MAX_FILE_BYTES:
            raise ValueError(f"Workspace file exceeds 32 MiB: {path.name}")

    before = path.lstat()
    check(before)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    with os.fdopen(os.open(path, flags), "rb") as source:
        opened = os.fstat(source.fileno())
        check(opened)
        if not os.path.samestat(before, opened):
            raise FileTransactionConflict(f"File changed while opening: {path.name}")
        data = source.read(_MAX_FILE_BYTES + 1)
        after = os.fstat(source.fileno())
    if (
        len(data) > _MAX_FILE_BYTES
        or opened.st_mtime_ns != after.st_mtime_ns
        or opened.st_size != after.st_size
    ):
        raise FileTransactionConflict(f"File changed while reading: {path.name}")
    return data


def _walk(root: Path, workspace: Workspace | None = None):
    count = 0
    for directory, folders, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        count += len(folders) + len(filenames)
        if count > _MAX_ENTRIES:
            raise ValueError("Workspace scan exceeds 20,000 entries")
        folders[:] = [
            name
            for name in sorted(folders)
            if name.casefold() not in _PRIVATE_DIRS
            and not (current / name).is_symlink()
            and not _is_reparse_point((current / name).lstat())
            and (workspace is None or not workspace.is_ignored(current / name))
        ]
        for name in sorted(filenames):
            if name.casefold() in _PRIVATE_DIRS:
                continue
            path = current / name
            mode = path.lstat()
            if (
                not stat.S_ISREG(mode.st_mode)
                or _is_reparse_point(mode)
                or mode.st_nlink != 1
            ):
                continue
            if workspace is not None:
                try:
                    workspace.check(path)
                except PermissionError:
                    continue
            yield path


def prepare_workspace_copy(workspace: Workspace, thread_id: str, user_id: str) -> Path:
    destination = _copy_root(workspace, thread_id, user_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _copy_lock(destination):
        if destination.exists():
            return destination / "files"
        with tempfile.TemporaryDirectory(
            dir=destination.parent, prefix="stage-"
        ) as staging:
            stage = Path(staging) / "copy"
            files = stage / "files"
            files.mkdir(parents=True)
            baseline = {}
            size = 0
            for original in _walk(workspace.root, workspace):
                relative = original.relative_to(workspace.root)
                data = _read_regular(original)
                size += len(data)
                if size > _MAX_COPY_BYTES:
                    raise ValueError(
                        "Workspace copy exceeds 512 MiB; narrow the read scope"
                    )
                copy = files / relative
                copy.parent.mkdir(parents=True, exist_ok=True)
                copy.write_bytes(data)
                copy.chmod(stat.S_IMODE(original.stat().st_mode) & 0o777)
                baseline[relative.as_posix()] = hash_bytes(data)
            (stage / "baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
            stage.rename(destination)
    return destination / "files"


def workspace_copy_diff(
    workspace: Workspace, thread_id: str, user_id: str
) -> list[dict[str, str]]:
    root = _copy_root(workspace, thread_id, user_id)
    if not root.is_dir():
        return []
    with _copy_lock(root):
        baseline = json.loads((root / "baseline.json").read_text(encoding="utf-8"))
        current = {
            path.relative_to(root / "files").as_posix(): hash_bytes(_read_regular(path))
            for path in _walk(root / "files")
        }
        changes = []
        for path in sorted(baseline.keys() | current.keys()):
            if baseline.get(path) == current.get(path):
                continue
            status = (
                "deleted"
                if path not in current
                else "modified"
                if path in baseline
                else "added"
            )
            change = {"path": path, "status": status}
            if path in current:
                change["sha256"] = current[path]
            changes.append(change)
        return changes


def review_workspace_copy(
    workspace: Workspace,
    thread_id: str,
    user_id: str,
    relative: str,
) -> dict[str, str]:
    root = _copy_root(workspace, thread_id, user_id)
    original, staged, _ = _reviewed_file(workspace, root, relative)
    original = original or b""
    if (
        max(len(original), len(staged)) > 256 * 1024
        or b"\x00" in original
        or b"\x00" in staged
    ):
        return {
            "path": relative,
            "sha256": hash_bytes(staged),
            "diff": "[No text preview for binary files or files above 256 KiB]",
        }
    diff = difflib.unified_diff(
        original.decode("utf-8", errors="replace").splitlines(keepends=True),
        staged.decode("utf-8", errors="replace").splitlines(keepends=True),
        fromfile=f"a/{relative}",
        tofile=f"b/{relative}",
    )
    lines = list(islice(diff, 201))
    rendered = "".join(lines[:200])[:16_000]
    if len(lines) > 200 or len(rendered) == 16_000:
        rendered += "\n[preview truncated]"
    return {"path": relative, "sha256": hash_bytes(staged), "diff": rendered}


def _reviewed_file(
    workspace: Workspace, root: Path, relative: str
) -> tuple[bytes | None, bytes, Path]:
    path = Path(relative)
    if (
        not path.parts
        or path.anchor
        or ".." in path.parts
        or any(part.casefold() in _PRIVATE_DIRS or ":" in part for part in path.parts)
    ):
        raise PermissionError("Use a relative Workspace file path")
    target = workspace.check(path, write=True)
    source = Workspace.from_path(root / "files").resolve(path)
    data = _read_regular(source)
    try:
        previous = _read_regular(target)
    except FileNotFoundError:
        previous = None
    baseline = json.loads((root / "baseline.json").read_text(encoding="utf-8"))
    previous_hash = hash_bytes(previous) if previous is not None else None
    if previous_hash not in (baseline.get(path.as_posix()), hash_bytes(data)):
        raise FileTransactionConflict(
            f"Host file changed since the copy was made: {path.as_posix()}"
        )
    return previous, data, target


def apply_workspace_copy(
    workspace: Workspace,
    thread_id: str,
    user_id: str,
    relative: str,
    sha256: str,
) -> str:
    root = _copy_root(workspace, thread_id, user_id)
    if not root.is_dir():
        raise FileNotFoundError("This thread has no Docker Workspace copy")
    write_locks = get_paths().base_dir / "workspace-write-locks"
    write_locks.mkdir(parents=True, exist_ok=True)
    write_lock = write_locks / workspace.root_fingerprint
    with _copy_lock(write_lock), _copy_lock(root):
        previous, data, target = _reviewed_file(workspace, root, relative)
        if hash_bytes(data) != sha256:
            raise FileTransactionConflict(
                "Sandbox file changed since review; review it again"
            )
        relative = target.relative_to(workspace.root).as_posix()
        if data != previous:
            operation = str(uuid.uuid4())
            target.parent.mkdir(parents=True, exist_ok=True)
            store = FileTransactionStore(
                transaction_root(thread_id, workspace=workspace),
                thread_id,
                workspace=workspace,
            )
            prepared = store.prepare(
                operation_id=operation,
                tool_call_id=operation,
                tool_name="workspace_apply",
                display_path=relative,
                target_path=str(target),
                pre_content=previous,
                expected_content=data,
                staged_path=str(staged_content_path(target, operation)),
            )
            applied = store.continue_prepared(prepared)
            if applied.status != APPLIED:
                raise FileTransactionConflict(
                    f"Workspace apply did not complete: {relative}"
                )
            store.commit(operation)
        baseline_path = root / "baseline.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline[relative] = sha256
        _atomic_write_bytes(baseline_path, json.dumps(baseline).encode())
    return relative
