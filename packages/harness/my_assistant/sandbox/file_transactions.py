"""Crash-safe transaction log for local file mutations.

The filesystem and LangGraph checkpointer cannot share one database transaction.
This module bridges that gap with a small per-thread SQLite write-ahead log.
File tools persist PREPARED before replacing the target and APPLIED afterwards;
the client marks COMMITTED only after the matching ToolMessage is observable in a
materialized checkpoint.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from my_assistant.config.paths import get_paths
from my_assistant.config.workspace import Workspace
from my_assistant.utils.thread_id import validate_thread_id

PREPARED = "PREPARED"
APPLIED = "APPLIED"
COMMITTED = "COMMITTED"
CONFLICT = "CONFLICT"
ROLLBACK_PREPARED = "ROLLBACK_PREPARED"
ROLLED_BACK = "ROLLED_BACK"
ABANDONED = "ABANDONED"
FAILED = "FAILED"

ACTIVE_STATUSES = (PREPARED, APPLIED, CONFLICT, ROLLBACK_PREPARED)
RECOVERY_STATUSES = (*ACTIVE_STATUSES, ROLLED_BACK, ABANDONED)
_MISSING_HASH = "__missing__"
_DB_NAME = "file-transactions.sqlite3"
_SNAPSHOT_DIR = "file-transaction-snapshots"
_LOCKS: dict[str, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


class FileTransactionError(RuntimeError):
    """Base error for file transaction recovery."""


class FileTransactionConflict(FileTransactionError):
    """The target differs from both the before and expected versions."""


class FileTransactionRecoveryRequired(FileTransactionError):
    """A prepared operation must be explicitly continued or abandoned."""


@dataclass(frozen=True)
class FileTransaction:
    operation_id: str
    thread_id: str
    tool_call_id: str
    tool_name: str
    display_path: str
    target_path: str
    pre_hash: str
    pre_content: bytes | None
    expected_hash: str
    expected_content: bytes
    pre_exists: bool
    snapshot_path: str | None
    staged_path: str | None
    status: str
    result: str
    error: str | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("pre_content", None)
        value.pop("expected_content", None)
        return value


def _now() -> str:
    return datetime.now(UTC).isoformat()


def hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def hash_path(path: str | Path) -> str:
    candidate = Path(path)
    if not candidate.exists():
        return _MISSING_HASH
    if not candidate.is_file():
        raise FileTransactionConflict(f"File transaction target is not a regular file: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transaction_root(
    thread_id: str,
    *,
    user_id: str | None = None,
    workspace: object = None,
    thread_data: object = None,
) -> Path:
    """Return the private per-thread runtime root for transaction metadata."""
    thread_id = validate_thread_id(thread_id)
    resolved_workspace = workspace if isinstance(workspace, Workspace) else Workspace.from_context(workspace)
    if resolved_workspace is None and isinstance(thread_data, dict):
        resolved_workspace = Workspace.from_context(thread_data.get("workspace_policy"))
    if resolved_workspace is not None:
        root = resolved_workspace.runtime_dir(thread_id)
    else:
        root = get_paths().thread_dir(thread_id, user_id=user_id)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _lock_for(root: Path) -> threading.RLock:
    key = os.path.normcase(str(root.resolve()))
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


class FileTransactionStore:
    """Per-thread durable operation log with file snapshots for rollback."""

    def __init__(self, root: str | Path, thread_id: str, *, workspace: object = None):
        self.root = Path(root)
        self.thread_id = validate_thread_id(thread_id)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / _DB_NAME
        self.snapshot_dir = self.root / _SNAPSHOT_DIR
        self.workspace = workspace if isinstance(workspace, Workspace) else Workspace.from_context(workspace)
        self._lock = _lock_for(self.root)
        self._ensure_schema()

    def _validate_target(self, path: str) -> None:
        if self.workspace is not None:
            self.workspace.check(path, write=True)

    def _expected_staged_path(self, target_path: str, operation_id: str) -> Path:
        return Path(os.path.abspath(staged_content_path(target_path, operation_id)))

    def _validate_staged_path(self, transaction: FileTransaction) -> Path:
        expected = self._expected_staged_path(transaction.target_path, transaction.operation_id)
        actual = Path(os.path.abspath(transaction.staged_path)) if transaction.staged_path else None
        if actual != expected or expected.is_symlink():
            raise FileTransactionConflict(f"Invalid staged path in transaction {transaction.operation_id}")
        return expected

    def _expected_snapshot_path(self, operation_id: str) -> Path:
        token = hashlib.sha256(operation_id.encode()).hexdigest()
        return Path(os.path.abspath(self.snapshot_dir / f"{token}.before"))

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _ensure_schema(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS file_transactions (
                    operation_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    tool_call_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    display_path TEXT NOT NULL,
                    target_path TEXT NOT NULL,
                    pre_hash TEXT NOT NULL,
                    pre_content BLOB,
                    expected_hash TEXT NOT NULL,
                    expected_content BLOB NOT NULL DEFAULT X'',
                    pre_exists INTEGER NOT NULL,
                    snapshot_path TEXT,
                    staged_path TEXT,
                    status TEXT NOT NULL,
                    result TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(file_transactions)")}
            if "expected_content" not in columns:
                connection.execute("ALTER TABLE file_transactions ADD COLUMN expected_content BLOB NOT NULL DEFAULT X''")
            if "pre_content" not in columns:
                connection.execute("ALTER TABLE file_transactions ADD COLUMN pre_content BLOB")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_file_transactions_status ON file_transactions(thread_id, status, created_at)")

    @staticmethod
    def _from_row(row: sqlite3.Row) -> FileTransaction:
        values = dict(row)
        values["pre_exists"] = bool(values["pre_exists"])
        return FileTransaction(**values)

    def get(self, operation_id: str) -> FileTransaction | None:
        with self._lock, self._connect() as connection:
            row = connection.execute("SELECT * FROM file_transactions WHERE operation_id = ?", (operation_id,)).fetchone()
        return self._from_row(row) if row is not None else None

    def list(self, statuses: Iterable[str] | None = None) -> list[FileTransaction]:
        params: list[Any] = [self.thread_id]
        sql = "SELECT * FROM file_transactions WHERE thread_id = ?"
        if statuses:
            status_values = tuple(statuses)
            sql += f" AND status IN ({','.join('?' for _ in status_values)})"
            params.extend(status_values)
        sql += " ORDER BY created_at, operation_id"
        with self._lock, self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._from_row(row) for row in rows]

    def prepare(
        self,
        *,
        operation_id: str,
        tool_call_id: str,
        tool_name: str,
        display_path: str,
        target_path: str,
        pre_content: bytes | None,
        expected_content: bytes,
        staged_path: str,
        result: str = "OK",
    ) -> FileTransaction:
        self._validate_target(target_path)
        if Path(os.path.abspath(staged_path)) != self._expected_staged_path(target_path, operation_id):
            raise FileTransactionConflict(f"Invalid staged path for operation {operation_id}")
        existing = self.get(operation_id)
        pre_hash = _MISSING_HASH if pre_content is None else hash_bytes(pre_content)
        expected_hash = hash_bytes(expected_content)
        if existing is not None:
            identity = (existing.tool_call_id, existing.tool_name, os.path.normcase(existing.target_path), existing.pre_hash, existing.expected_hash)
            requested = (tool_call_id, tool_name, os.path.normcase(target_path), pre_hash, expected_hash)
            if identity != requested:
                raise FileTransactionConflict(f"Operation id {operation_id!r} was reused for a different file mutation")
            return existing

        snapshot_path: str | None = None
        if pre_content is not None:
            snapshot = self.snapshot_dir / f"{hashlib.sha256(operation_id.encode()).hexdigest()}.before"
            snapshot_path = str(snapshot)

        timestamp = _now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO file_transactions (
                    operation_id, thread_id, tool_call_id, tool_name, display_path,
                    target_path, pre_hash, pre_content, expected_hash, expected_content, pre_exists, snapshot_path,
                    staged_path, status, result, error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    operation_id,
                    self.thread_id,
                    tool_call_id,
                    tool_name,
                    display_path,
                    target_path,
                    pre_hash,
                    pre_content,
                    expected_hash,
                    expected_content,
                    int(pre_content is not None),
                    snapshot_path,
                    staged_path,
                    PREPARED,
                    result,
                    timestamp,
                    timestamp,
                ),
            )
        prepared = self.get(operation_id)
        if prepared is None:
            raise FileTransactionError(f"Failed to persist file transaction {operation_id}")
        self._ensure_snapshot(prepared)
        return self.get(operation_id) or prepared

    def _ensure_snapshot(self, transaction: FileTransaction) -> None:
        if not transaction.pre_exists:
            return
        expected_snapshot = self._expected_snapshot_path(transaction.operation_id)
        if transaction.snapshot_path and Path(os.path.abspath(transaction.snapshot_path)) != expected_snapshot:
            raise FileTransactionConflict(f"Invalid rollback snapshot path in transaction {transaction.operation_id}")
        if expected_snapshot.is_symlink():
            raise FileTransactionConflict(f"Rollback snapshot was replaced by a symlink for {transaction.operation_id}")
        if expected_snapshot.is_file():
            return
        if transaction.pre_content is None or hash_bytes(transaction.pre_content) != transaction.pre_hash:
            raise FileTransactionError(f"Rollback snapshot is unavailable for {transaction.operation_id}")
        snapshot = expected_snapshot
        _atomic_write_bytes(snapshot, transaction.pre_content)
        if not transaction.snapshot_path:
            with self._lock, self._connect() as connection:
                connection.execute(
                    "UPDATE file_transactions SET snapshot_path = ?, updated_at = ? WHERE operation_id = ?",
                    (str(snapshot), _now(), transaction.operation_id),
                )

    def transition(self, operation_id: str, status: str, *, error: str | None = None) -> FileTransaction:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE file_transactions SET status = ?, error = ?, updated_at = ? WHERE operation_id = ?",
                (status, error, _now(), operation_id),
            )
        value = self.get(operation_id)
        if value is None:
            raise FileTransactionError(f"Unknown file transaction: {operation_id}")
        return value

    def reconcile(self, transaction: FileTransaction) -> FileTransaction:
        self._validate_target(transaction.target_path)
        current_hash = hash_path(transaction.target_path)
        if transaction.status == ROLLBACK_PREPARED:
            if current_hash == transaction.pre_hash:
                return self.transition(transaction.operation_id, ROLLED_BACK)
            if current_hash == transaction.expected_hash:
                return transaction
            return self.transition(
                transaction.operation_id,
                CONFLICT,
                error="Target changed while rollback was in progress; refusing automatic overwrite",
            )
        if current_hash == transaction.expected_hash:
            if transaction.status == PREPARED:
                return self.transition(transaction.operation_id, APPLIED)
            return transaction
        if current_hash == transaction.pre_hash:
            if transaction.status == APPLIED:
                return self.transition(transaction.operation_id, PREPARED)
            return transaction
        detail = "Target changed after the transaction was prepared; refusing automatic overwrite"
        if transaction.status != CONFLICT or transaction.error != detail:
            return self.transition(transaction.operation_id, CONFLICT, error=detail)
        return transaction

    def reconcile_active(self) -> list[FileTransaction]:
        return [self.reconcile(item) for item in self.list(ACTIVE_STATUSES)]

    def continue_prepared(self, transaction: FileTransaction) -> FileTransaction:
        self._validate_target(transaction.target_path)
        transaction = self.reconcile(transaction)
        if transaction.status == APPLIED:
            return transaction
        if transaction.status == CONFLICT:
            raise FileTransactionConflict(transaction.error or f"Conflicting file transaction: {transaction.operation_id}")
        if transaction.status != PREPARED:
            return transaction
        if hash_path(transaction.target_path) != transaction.pre_hash:
            raise FileTransactionConflict(f"Target changed before operation {transaction.operation_id} could continue")
        self._ensure_snapshot(transaction)
        staged = self._validate_staged_path(transaction)
        if not staged.is_file():
            if hash_bytes(transaction.expected_content) != transaction.expected_hash:
                return self.transition(transaction.operation_id, CONFLICT, error="Prepared content is unavailable; cannot continue")
            _atomic_write_bytes(staged, transaction.expected_content)
        _replace_staged_file(staged, Path(transaction.target_path))
        if hash_path(transaction.target_path) != transaction.expected_hash:
            return self.transition(transaction.operation_id, CONFLICT, error="Atomic replacement did not produce the expected hash")
        return self.transition(transaction.operation_id, APPLIED)

    def commit(self, operation_id: str) -> FileTransaction:
        transaction = self.get(operation_id)
        if transaction is None:
            raise FileTransactionError(f"Unknown file transaction: {operation_id}")
        if transaction.status == COMMITTED:
            return transaction
        if transaction.status != APPLIED:
            raise FileTransactionError(f"Cannot commit {operation_id} from {transaction.status}")
        value = self.transition(operation_id, COMMITTED)
        _unlink_if_exists(str(self._validate_staged_path(transaction)))
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE file_transactions SET expected_content = X'' WHERE operation_id = ?",
                (operation_id,),
            )
        return value

    def rollback(self, operation_id: str) -> FileTransaction:
        transaction = self.get(operation_id)
        if transaction is None:
            raise FileTransactionError(f"Unknown file transaction: {operation_id}")
        self._validate_target(transaction.target_path)
        if transaction.status == ROLLED_BACK:
            return transaction
        if transaction.status == ROLLBACK_PREPARED:
            transaction = self.reconcile(transaction)
            if transaction.status == ROLLED_BACK:
                return transaction
        current_hash = hash_path(transaction.target_path)
        if current_hash not in {transaction.pre_hash, transaction.expected_hash}:
            raise FileTransactionConflict(f"Refusing rollback because {transaction.display_path} was modified outside this transaction")
        transaction = self.transition(operation_id, ROLLBACK_PREPARED)
        target = Path(transaction.target_path)
        if transaction.pre_exists:
            self._ensure_snapshot(transaction)
            transaction = self.get(operation_id) or transaction
            _atomic_write_bytes(target, Path(transaction.snapshot_path).read_bytes())
        elif target.exists():
            target.unlink()
            _fsync_directory(target.parent)
        _unlink_if_exists(str(self._validate_staged_path(transaction)))
        return self.transition(operation_id, ROLLED_BACK)

    def abandon(self, operation_id: str) -> FileTransaction:
        transaction = self.get(operation_id)
        if transaction is None:
            raise FileTransactionError(f"Unknown file transaction: {operation_id}")
        if transaction.status == COMMITTED:
            raise FileTransactionError(f"Cannot abandon committed operation {operation_id}; roll it back instead")
        staged = self._validate_staged_path(transaction)
        value = self.transition(operation_id, ABANDONED)
        _unlink_if_exists(str(staged))
        return value


def stage_expected_content(target: str | Path, operation_id: str, content: bytes) -> Path:
    staged = staged_content_path(target, operation_id)
    staged.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(staged, content)
    return staged


def staged_content_path(target: str | Path, operation_id: str) -> Path:
    target_path = Path(target)
    token = hashlib.sha256(operation_id.encode("utf-8")).hexdigest()[:20]
    return target_path.parent / f".my-assistant-write-{token}.tmp"


def _replace_staged_file(staged: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, target)
    _fsync_directory(target.parent)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_if_exists(path: str | None) -> None:
    if not path:
        return
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
