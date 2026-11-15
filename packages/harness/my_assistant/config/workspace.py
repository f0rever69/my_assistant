"""Run-scoped workspace policy for the embedded CLI client."""

from __future__ import annotations

import fnmatch
import hashlib
import os
import stat
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from my_assistant.utils.thread_id import validate_thread_id

DEFAULT_IGNORED_PATTERNS = (
    ".git/**",
    ".my-assistant/**",
    "**/node_modules/**",
    "**/__pycache__/**",
    "**/.env",
    "**/.env.*",
    "**/*.pem",
    "**/*.key",
    "**/.ssh/**",
    "**/.ssh",
    "**/.aws/**",
    "**/.azure/**",
    "**/.config/gcloud/**",
    "**/.netrc",
    "**/.npmrc",
    "**/.pypirc",
    "**/credentials",
    "**/credentials.*",
)

_current_workspace: ContextVar[Workspace | None] = ContextVar("my_assistant_workspace", default=None)


@dataclass(frozen=True)
class Workspace:
    id: str
    name: str
    root_path: str
    allowed_read_paths: tuple[str, ...] = (".",)
    allowed_write_paths: tuple[str, ...] = (".",)
    ignored_patterns: tuple[str, ...] = DEFAULT_IGNORED_PATTERNS
    shell_enabled: bool = False
    network_enabled: bool = False
    root_fingerprint: str = ""
    registry_managed: bool = False

    @property
    def root(self) -> Path:
        return Path(self.root_path).resolve()

    def runtime_dir(self, thread_id: str) -> Path:
        return self.resolve(Path(".my-assistant") / "threads" / validate_thread_id(thread_id))

    def ensure_runtime_dirs(self, thread_id: str) -> dict[str, str]:
        runtime = self.runtime_dir(thread_id)
        paths = {
            "uploads_path": self.resolve(runtime / "uploads"),
            "outputs_path": self.resolve(runtime / "outputs"),
            "acp_workspace_path": self.resolve(runtime / "acp-workspace"),
        }
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        return {key: str(path) for key, path in paths.items()}

    def resolve(self, path: str | os.PathLike[str]) -> Path:
        """Resolve a workspace path without traversing links or reparse points."""
        if self.root_fingerprint and fingerprint_root(self.root_path) != self.root_fingerprint:
            raise PermissionError("Workspace root identity changed; file access is unavailable")
        candidate = Path(path)
        root = self.root
        lexical = Path(os.path.abspath(candidate if candidate.is_absolute() else root / candidate))
        try:
            relative = lexical.relative_to(root)
        except ValueError as exc:
            raise PermissionError(f"Path is outside workspace: {path}") from exc

        current = root
        for part in relative.parts:
            current = current / part
            try:
                info = current.lstat()
            except FileNotFoundError:
                continue
            if current.is_symlink() or _is_reparse_point(info):
                raise PermissionError(f"Workspace paths may not traverse symlinks or reparse points: {path}")
        return lexical

    def _matches_scope(self, relative: str, scopes: tuple[str, ...]) -> bool:
        relative = relative.replace("\\", "/")
        while relative.startswith("./"):
            relative = relative[2:]
        for scope in scopes:
            normalized = str(scope).replace("\\", "/").strip("/") or "."
            if normalized == "." or relative == normalized or relative.startswith(normalized + "/"):
                return True
            if fnmatch.fnmatch(relative, normalized):
                return True
        return False

    def is_ignored(self, path: str | os.PathLike[str]) -> bool:
        resolved = self.resolve(path)
        relative = resolved.relative_to(self.root).as_posix()
        return any(
            fnmatch.fnmatch(relative, pattern)
            or fnmatch.fnmatch(relative, pattern.removeprefix("**/"))
            for pattern in (*DEFAULT_IGNORED_PATTERNS, *self.ignored_patterns)
        )

    def check(self, path: str | os.PathLike[str], *, write: bool = False) -> Path:
        resolved = self.resolve(path)
        relative = resolved.relative_to(self.root).as_posix()
        scopes = self.allowed_write_paths if write else self.allowed_read_paths
        if not self._matches_scope(relative, scopes):
            action = "write" if write else "read"
            raise PermissionError(f"Workspace does not allow {action} access: {path}")
        # Agent-visible outputs live under .my-assistant, while the rest of
        # that directory contains private checkpoints and runtime metadata.
        relative_parts = resolved.relative_to(self.root).parts
        runtime_category = (
            len(relative_parts) >= 4
            and relative_parts[0] == ".my-assistant"
            and relative_parts[1] == "threads"
            and relative_parts[3] in {"outputs", "uploads"}
        )
        if runtime_category:
            try:
                validate_thread_id(relative_parts[2])
            except ValueError:
                runtime_category = False
        if runtime_category and (relative_parts[3] == "outputs" or not write):
            return resolved
        if self.is_ignored(resolved):
            raise PermissionError(f"Workspace path is protected: {path}")
        return resolved

    @classmethod
    def from_path(cls, value: str | os.PathLike[str]) -> Workspace:
        root = Path(value).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"Workspace root is not a directory: {root}")
        root_text = str(root)
        digest = hashlib.sha256(os.path.normcase(root_text).encode("utf-8")).hexdigest()[:16]
        return cls(id=f"ws-{digest}", name=root.name or root_text, root_path=root_text, root_fingerprint=fingerprint_root(root))

    def to_context(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("allowed_read_paths", "allowed_write_paths", "ignored_patterns"):
            value[key] = list(value[key])
        return value

    @classmethod
    def from_context(cls, value: object) -> Workspace | None:
        if not isinstance(value, dict):
            return None
        try:
            if any(not isinstance(value.get(key), str) or not value[key].strip() for key in ("id", "name", "root_path")):
                return None
            root = Path(str(value["root_path"])).resolve(strict=True)
            if not root.is_dir():
                return None
            fingerprint = fingerprint_root(root)
            if value.get("root_fingerprint") and value["root_fingerprint"] != fingerprint:
                return None
            scopes = {}
            for key, default in (("allowed_read_paths", (".",)), ("allowed_write_paths", (".",)), ("ignored_patterns", DEFAULT_IGNORED_PATTERNS)):
                raw = value.get(key, default)
                if not isinstance(raw, (tuple, list)) or any(not isinstance(item, str) or not item for item in raw):
                    return None
                if key != "ignored_patterns" and any(Path(item).is_absolute() or ":" in item or ".." in item.replace("\\", "/").split("/") for item in raw):
                    return None
                scopes[key] = tuple(raw)
            return cls(
                id=str(value["id"]),
                name=str(value["name"]),
                root_path=str(root),
                **scopes,
                shell_enabled=value.get("shell_enabled") is True,
                network_enabled=value.get("network_enabled") is True,
                root_fingerprint=fingerprint,
                registry_managed=value.get("registry_managed") is True,
            )
        except (KeyError, OSError, TypeError, ValueError):
            return None


def fingerprint_root(path: str | os.PathLike[str]) -> str:
    """Bind to the canonical path AND directory identity, not its mutable mtime."""
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Workspace root is not a directory")
    info = root.stat()
    identity = f"{os.path.normcase(str(root))}\0{info.st_dev}\0{info.st_ino}\0{getattr(info, 'st_birthtime_ns', '')}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _is_reparse_point(info: os.stat_result) -> bool:
    """Detect Windows junctions and other reparse points without following them."""
    attributes = getattr(info, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def get_current_workspace() -> Workspace | None:
    return _current_workspace.get()


@contextmanager
def workspace_scope(value: object):
    workspace = value if isinstance(value, Workspace) else Workspace.from_context(value)
    token = _current_workspace.set(workspace)
    try:
        yield workspace
    finally:
        _current_workspace.reset(token)
