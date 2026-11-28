"""Stage searchable Workspace files for Docker and apply reviewed results on the host."""

import difflib
import hashlib
import json
import os
import sqlite3
import stat
import tempfile
import time
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
    _fsync_directory,
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


def _publish_workspace_copy(stage: Path, destination: Path) -> None:
    """Retry short-lived Windows locks while preserving atomic publication."""
    for delay in (0.05, 0.1, 0.2, 0.4, None):
        try:
            stage.rename(destination)
            return
        except OSError as exc:
            if delay is None or getattr(exc, "winerror", None) not in {5, 32, 33} or destination.exists():
                raise
            time.sleep(delay)


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
            _publish_workspace_copy(stage, destination)
    return destination / "files"


def _refresh_snapshot(root: Path, workspace: Workspace | None = None) -> dict[str, str]:
    snapshot = {}
    size = 0
    for path in _walk(root, workspace):
        data = _read_regular(path)
        size += len(data)
        if size > _MAX_COPY_BYTES:
            raise ValueError("Workspace refresh exceeds 512 MiB; narrow the read scope")
        snapshot[path.relative_to(root).as_posix()] = hash_bytes(data)
    return snapshot


def _refresh_baseline(root: Path, workspace: Workspace) -> dict[str, str]:
    baseline = json.loads(_read_regular(root / "baseline.json"))
    if not isinstance(baseline, dict):
        raise TypeError("Invalid Docker Workspace baseline")
    for name, digest in baseline.items():
        if not isinstance(name, str) or not isinstance(digest, str):
            raise TypeError("Invalid Docker Workspace baseline entry")
        path = Path(name)
        if (
            not path.parts
            or path.anchor
            or path.as_posix() != name
            or "\\" in name
            or ".." in path.parts
            or any(part.casefold() in _PRIVATE_DIRS or ":" in part for part in path.parts)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Invalid Docker Workspace baseline path or hash")
        workspace.check(name)
    return baseline


def _refresh_content(workspace: Workspace, name: str) -> bytes | None:
    path = workspace.check(name)
    current = workspace.root
    for part in path.relative_to(current).parts[:-1]:
        current = current / part
        if current.exists() and not current.is_dir():
            raise PermissionError(f"Workspace path parent is not a directory: {name}")
    try:
        return _read_regular(path)
    except FileNotFoundError:
        return None


def _content_hash(content: bytes | None) -> str | None:
    return hash_bytes(content) if content is not None else None


def _check_refresh_versions(
    workspace: Workspace,
    staged: Workspace,
    name: str,
    host_hash: str | None,
    staged_hash: str | None,
) -> bytes | None:
    data = _refresh_content(workspace, name)
    if _content_hash(data) != host_hash:
        raise FileTransactionConflict(f"Host file changed during refresh: {name}")
    if _content_hash(_refresh_content(staged, name)) != staged_hash:
        raise FileTransactionConflict(f"Sandbox file changed during refresh: {name}")
    return data


def _write_refresh_file(
    staged: Workspace, name: str, data: bytes, expected_hash: str | None, mode: int
) -> None:
    target = staged.resolve(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    # An unpredictable, exclusively created temporary avoids following a
    # sandbox-created link at the temporary filename used by a generic writer.
    descriptor, temporary = tempfile.mkstemp(prefix=".refresh-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        target = staged.resolve(name)
        if _content_hash(_refresh_content(staged, name)) != expected_hash:
            raise FileTransactionConflict(f"Sandbox file changed during refresh: {name}")
        os.chmod(temporary, mode & 0o777)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def refresh_workspace_copy(
    workspace: Workspace,
    thread_id: str,
    user_id: str,
    *,
    dry_run: bool = False,
) -> dict:
    """Pull host changes into an idle copy using its last synchronized baseline.

    No host file is written. Divergent edits and unsafe/type-changing paths
    reject the entire preflight, including unrelated changes. Copy/apply locks
    serialize host-side operations; callers must keep agents and other writers
    idle. Hashes are rechecked before writes, but arbitrary shell writers do not
    participate in these locks. Individual writes are atomic and the directory
    mounted by Docker is never replaced. After an interrupted batch, retrying
    recognizes already synchronized content without losing sandbox-only edits.
    """
    root = _copy_root(workspace, thread_id, user_id)
    if not root.is_dir():
        raise FileNotFoundError("This thread has no Docker Workspace copy")
    write_locks = get_paths().base_dir / "workspace-write-locks"
    write_locks.mkdir(parents=True, exist_ok=True)
    with _copy_lock(write_locks / workspace.root_fingerprint), _copy_lock(root):
        files = root / "files"
        info = files.lstat()
        if not stat.S_ISDIR(info.st_mode) or _is_reparse_point(info):
            raise PermissionError("Docker Workspace copy must be a regular directory")
        staged = Workspace.from_path(files)
        baseline = _refresh_baseline(root, workspace)
        host = _refresh_snapshot(workspace.root, workspace)
        sandbox = _refresh_snapshot(files)
        changes, preserved, conflicts = [], [], []
        for name in sorted(baseline.keys() | host.keys() | sandbox.keys()):
            before, local, current = baseline.get(name), host.get(name), sandbox.get(name)
            if local == before:
                if current != before:
                    preserved.append(name)
                continue
            if current != before and current != local:
                conflicts.append({"path": name, "reason": "Host and sandbox changed differently"})
                continue
            try:
                _check_refresh_versions(workspace, staged, name, local, current)
            except (OSError, ValueError, FileTransactionConflict) as exc:
                conflicts.append({"path": name, "reason": str(exc)})
                continue
            status = (
                "aligned" if current == local else
                "deleted" if local is None else
                "added" if before is None else "modified"
            )
            changes.append({"path": name, "status": status})
        result = {"applied": False, "changes": changes, "preserved": preserved, "conflicts": conflicts}
        if conflicts or dry_run:
            return result
        # Validate every candidate again before the first mutation. Each write
        # additionally checks its own versions so stale scans never bless edits.
        for change in changes:
            name = change["path"]
            _check_refresh_versions(workspace, staged, name, host.get(name), sandbox.get(name))
        updated = dict(baseline)
        for change in changes:
            name = change["path"]
            data = _check_refresh_versions(workspace, staged, name, host.get(name), sandbox.get(name))
            if change["status"] == "deleted":
                staged.resolve(name).unlink()
            elif change["status"] != "aligned":
                mode = stat.S_IMODE(workspace.check(name).stat().st_mode)
                _write_refresh_file(staged, name, data, sandbox.get(name), mode)
            if name in host:
                updated[name] = host[name]
            else:
                updated.pop(name, None)
        if updated != baseline:
            _atomic_write_bytes(root / "baseline.json", json.dumps(updated).encode())
        result["applied"] = True
        return result


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
        _record_sync(root, relative, sha256, "host_updated")
    return relative


def _record_sync(root: Path, relative: str, digest: str | None, status: str) -> None:
    path = root / "sync-status.json"
    records = json.loads(_read_regular(path)) if path.exists() else {}
    if not isinstance(records, dict):
        raise TypeError("Invalid Workspace synchronization record")
    records.pop(relative, None)
    records[relative] = {"sha256": digest, "status": status}
    _atomic_write_bytes(path, json.dumps(dict(list(records.items())[-256:])).encode())


def _copy_workspace(root: Path) -> Workspace:
    files = root / "files"
    info = files.lstat()
    if not stat.S_ISDIR(info.st_mode) or _is_reparse_point(info):
        raise PermissionError("Docker Workspace copy must be a regular directory")
    return Workspace.from_path(files)


def workspace_copy_status(workspace: Workspace, thread_id: str, user_id: str) -> dict:
    """Read current bytes, rather than interpreting a model's completion claim."""
    root = _copy_root(workspace, thread_id, user_id)
    if not root.is_dir():
        return {"mode": "sandbox_workspace_copy", "files": [], "copy_exists": False}
    with _copy_lock(root):
        staged = _copy_workspace(root)
        baseline = _refresh_baseline(root, workspace)
        host = _refresh_snapshot(workspace.root, workspace)
        sandbox = _refresh_snapshot(root / "files")
        sync_path = root / "sync-status.json"
        records = json.loads(_read_regular(sync_path)) if sync_path.exists() else {}
        if not isinstance(records, dict):
            raise TypeError("Invalid Workspace synchronization record")
        files = []
        for name in sorted(baseline.keys() | host.keys() | sandbox.keys() | records.keys()):
            # A sandbox-created private filename must never expose host secrets.
            try:
                workspace.check(name)
                staged.resolve(name)
            except PermissionError:
                continue
            before, local, current = baseline.get(name), host.get(name), sandbox.get(name)
            record = records.get(name)
            if current != before and current != local:
                status = "conflict" if local != before else "pending_apply"
            elif local != before and current == before:
                status = "local_pending_refresh"
            elif isinstance(record, dict) and record.get("sha256") == local == current:
                status = record.get("status")
                if status not in {"host_updated", "host_deleted", "local_kept"}:
                    continue
            elif current == local and current != before:
                status = "host_matches"
            else:
                continue
            files.append({
                "path": name, "status": status, "host_sha256": local,
                "sandbox_sha256": current, "baseline_sha256": before,
            })
        return {"mode": "sandbox_workspace_copy", "files": files, "copy_exists": True}


def _conflict_review(workspace: Workspace, root: Path, relative: str, thread_id: str, user_id: str) -> tuple[dict, bytes | None, bytes | None]:
    relative = Path(relative).as_posix()
    if not Path(relative).parts or Path(relative).anchor or ".." in Path(relative).parts or "\\" in relative or ":" in relative:
        raise PermissionError("Use a relative Workspace file path")
    if any(part.casefold() in _PRIVATE_DIRS for part in Path(relative).parts):
        raise PermissionError("Protected Workspace path")
    staged = _copy_workspace(root)
    baseline = _refresh_baseline(root, workspace)
    host, sandbox = _refresh_content(workspace, relative), _refresh_content(staged, relative)
    review = _review_versions(workspace, thread_id, user_id, {
        "path": relative, "host_sha256": _content_hash(host),
        "sandbox_sha256": _content_hash(sandbox), "baseline_sha256": baseline.get(relative),
    })
    return review, host, sandbox


def _review_versions(workspace: Workspace, thread_id: str, user_id: str, versions: dict) -> dict:
    review = {key: versions[key] for key in ("path", "host_sha256", "sandbox_sha256", "baseline_sha256")}
    identity = [workspace.to_context(), thread_id, user_id, review]
    review["resolution_token"] = hash_bytes(json.dumps(identity, sort_keys=True).encode())
    review["conflict"] = review["host_sha256"] != review["sandbox_sha256"] and all(
        review[key] != review["baseline_sha256"] for key in ("host_sha256", "sandbox_sha256")
    )
    return review


def review_workspace_conflict(workspace: Workspace, thread_id: str, user_id: str, relative: str) -> dict:
    """Review both versions even when ordinary apply is correctly blocked."""
    root = _copy_root(workspace, thread_id, user_id)
    if not root.is_dir():
        raise FileNotFoundError("This thread has no Docker Workspace copy")
    with _copy_lock(root):
        review, host, sandbox = _conflict_review(workspace, root, relative, thread_id, user_id)
        if any(data is not None and (len(data) > 256 * 1024 or b"\x00" in data) for data in (host, sandbox)):
            review["diff"] = "[No text preview for binary files or files above 256 KiB]"
        else:
            lines = difflib.unified_diff(
                (host or b"").decode("utf-8", errors="replace").splitlines(keepends=True),
                (sandbox or b"").decode("utf-8", errors="replace").splitlines(keepends=True),
                fromfile=f"local/{review['path']}", tofile=f"sandbox/{review['path']}",
            )
            review["diff"] = "".join(islice(lines, 201))[:16000]
        return review


def resolve_workspace_conflict(
    workspace: Workspace, thread_id: str, user_id: str, relative: str,
    choice: str, resolution_token: str,
) -> dict:
    """Apply a trusted user's version choice, bound to both bytes and baseline.

    Both versions are backed up outside the container mount before mutation.
    Choosing sandbox explicitly authorizes overwriting/deleting this one host
    path. File writes use the existing recoverable host transaction machinery.
    """
    if choice not in {"local", "sandbox"}:
        raise ValueError("Choose local or sandbox after reviewing /workspace conflicts <path>")
    root = _copy_root(workspace, thread_id, user_id)
    if not root.is_dir():
        raise FileNotFoundError("This thread has no Docker Workspace copy")
    locks = get_paths().base_dir / "workspace-write-locks"
    locks.mkdir(parents=True, exist_ok=True)
    with _copy_lock(locks / workspace.root_fingerprint), _copy_lock(root):
        return _apply_reviewed_version_locked(workspace, root, thread_id, user_id, relative, choice, resolution_token)


def _apply_reviewed_version_locked(
    workspace: Workspace, root: Path, thread_id: str, user_id: str, relative: str,
    choice: str, resolution_token: str,
    *, batch: bool = False,
) -> dict:
    """Apply one version while the caller holds the Workspace and copy locks."""
    review, host, sandbox = _conflict_review(workspace, root, relative, thread_id, user_id)
    if review["resolution_token"] != resolution_token:
        raise FileTransactionConflict("A file or baseline changed since review; review the conflict again")
    relative = review["path"]
    target = workspace.check(relative, write=(choice == "sandbox"))
    staged = _copy_workspace(root)
    operation = uuid.uuid4().hex
    backup = Workspace.from_path(root).resolve(Path("review-backups" if batch else "conflict-backups") / operation)
    backup.mkdir(parents=True)
    for label, data in (("local", host), ("sandbox", sandbox)):
        if data is not None:
            _atomic_write_bytes(backup / f"{label}.bin", data)
    journal = {**review, "choice": choice, "state": "prepared"}
    _atomic_write_bytes(backup / "resolution.json", json.dumps(journal).encode())
    # Re-read after backups so a newer version cannot use the old review.
    current, _, _ = _conflict_review(workspace, root, relative, thread_id, user_id)
    if current["resolution_token"] != resolution_token:
        raise FileTransactionConflict("A file changed while preparing resolution; review it again")
    if choice == "local":
        if host is None:
            if sandbox is not None:
                staged.resolve(relative).unlink()
        elif host != sandbox:
            _write_refresh_file(staged, relative, host, review["sandbox_sha256"], stat.S_IMODE(target.stat().st_mode))
        selected, status = host, "local_kept"
    elif sandbox is None:
        if host is not None:
            _check_refresh_versions(workspace, staged, relative, review["host_sha256"], None)
            target = workspace.check(relative, write=True)
            target.unlink()
            _fsync_directory(target.parent)
        selected, status = None, "host_deleted"
    else:
        if host != sandbox:
            store = FileTransactionStore(transaction_root(thread_id, workspace=workspace), thread_id, workspace=workspace)
            target.parent.mkdir(parents=True, exist_ok=True)
            transaction = store.prepare(
                operation_id=operation, tool_call_id=operation,
                tool_name="workspace_review" if batch else "workspace_resolve",
                display_path=relative, target_path=str(target), pre_content=host,
                expected_content=sandbox, staged_path=str(staged_content_path(target, operation)),
            )
            if store.continue_prepared(transaction).status != APPLIED:
                raise FileTransactionConflict(f"Workspace write-back did not complete: {relative}")
            store.commit(operation)
        selected, status = sandbox, "host_updated"
    digest = _content_hash(selected)
    baseline = _refresh_baseline(root, workspace)
    if digest is None:
        baseline.pop(relative, None)
    else:
        baseline[relative] = digest
    _atomic_write_bytes(root / "baseline.json", json.dumps(baseline).encode())
    _record_sync(root, relative, digest, status)
    _atomic_write_bytes(backup / "resolution.json", json.dumps({**journal, "state": "applied"}).encode())
    return {"path": relative, "choice": choice, "status": status, "backup_path": str(backup), "sha256": digest}
