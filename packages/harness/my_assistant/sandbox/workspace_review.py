"""Version-pinned document comparison for a user's Workspace review."""

import difflib
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from my_assistant.config.paths import get_paths
from my_assistant.config.workspace import Workspace
from my_assistant.sandbox.file_transactions import (
    FileTransactionConflict,
    FileTransactionError,
    _atomic_write_bytes,
)
from my_assistant.sandbox.workspace_copy import (
    _apply_reviewed_version_locked,
    _conflict_review,
    _copy_lock,
    _copy_root,
    _read_regular,
    _review_versions,
    workspace_copy_status,
)

_PREVIEW_CHARS = 16_000
_PREVIEW_LINES = 200
_MAX_DIFF_CHARS = 256 * 1024
_MAX_DIFF_LINES = 4_000


@dataclass(frozen=True)
class FileComparison:
    path: str
    original: Path | None
    candidate: Path | None
    original_size: int | None
    candidate_size: int | None
    text_preview: str
    is_text: bool


@dataclass(frozen=True)
class WorkspaceReviewChange:
    path: str
    kind: str
    token: str
    blocked_reason: str = ""


def list_workspace_review(workspace: Workspace, thread_id: str, user_id: str) -> list[WorkspaceReviewChange]:
    status = workspace_copy_status(workspace, thread_id, user_id)
    changes = []
    for item in status["files"]:
        if item["status"] not in {"pending_apply", "conflict"}:
            continue
        review = _review_versions(workspace, thread_id, user_id, item)
        reason = "两边已修改，请单独解决冲突" if review["conflict"] else ""
        try:
            workspace.check(review["path"], write=True)
        except PermissionError:
            reason = "工作区策略禁止写入"
        if review["sandbox_sha256"] is None:
            kind = "deleted"
        elif review["host_sha256"] is None:
            kind = "added"
        else:
            kind = "modified"
        changes.append(WorkspaceReviewChange(review["path"], kind, review["resolution_token"], reason))
    return changes


def apply_workspace_review(
    workspace: Workspace, thread_id: str, user_id: str, selected: Mapping[str, str]
) -> dict:
    """Preflight every approved version, then apply selected changes one at a time."""
    if not selected:
        raise ValueError("请先勾选要写回的文件。")
    root = _copy_root(workspace, thread_id, user_id)
    if not root.is_dir():
        raise FileNotFoundError("当前会话还没有沙箱副本。")
    locks = get_paths().base_dir / "workspace-write-locks"
    locks.mkdir(parents=True, exist_ok=True)
    with _copy_lock(locks / workspace.root_fingerprint), _copy_lock(root):
        targets = set()
        for path, token in selected.items():
            review, _, _ = _conflict_review(workspace, root, path, thread_id, user_id)
            if review["resolution_token"] != token:
                raise FileTransactionConflict(f"{path} 的版本已变化，请重新审阅。")
            host, sandbox, baseline = (review[field] for field in ("host_sha256", "sandbox_sha256", "baseline_sha256"))
            if host != baseline or sandbox == baseline:
                raise FileTransactionConflict(f"{path} 不是可批量写回的沙箱变更，请重新审阅。")
            target = workspace.check(review["path"], write=True)
            if target in targets:
                raise ValueError(f"同一个目标文件被重复选中：{path}")
            targets.add(target)

        results = []
        for path, token in selected.items():
            try:
                results.append(_apply_reviewed_version_locked(
                    workspace, root, thread_id, user_id, path, "sandbox", token, batch=True,
                ))
            except (OSError, ValueError, sqlite3.Error, FileTransactionError) as exc:
                return {
                    "applied": results, "failed_path": path, "error": str(exc),
                    "not_attempted": list(selected)[len(results) + 1 :],
                }
    return {"applied": results, "failed_path": None, "error": None, "not_attempted": []}


def _preview_text(data: bytes | None, suffix: str) -> str | None:
    if data is None:
        return ""
    if suffix == ".docx":
        from my_assistant.rag.documents import DocumentError, extract_passages

        try:
            return "\n".join(passage.text for passage in extract_passages(suffix, data))
        except DocumentError:
            return None
    if b"\x00" in data or suffix in {".doc", ".pdf", ".xlsx", ".pptx", ".odt"}:
        return None
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def _comparison_preview(host: bytes | None, sandbox: bytes | None, suffix: str) -> tuple[str, bool]:
    original, candidate = _preview_text(host, suffix), _preview_text(sandbox, suffix)
    is_text = suffix != ".docx" and original is not None and candidate is not None
    if original is None or candidate is None:
        return "此文件没有文本预览。请打开完整版本查看。", is_text
    if max(len(original), len(candidate)) > _MAX_DIFF_CHARS:
        return "文件内容超出终端预览范围。请打开完整版本对照。", is_text
    before, after = original.splitlines(), candidate.splitlines()
    if max(len(before), len(after)) > _MAX_DIFF_LINES:
        return "文件内容超出终端预览范围。请打开完整版本对照。", is_text
    diff = difflib.unified_diff(before, after, fromfile="原稿", tofile="候选稿", n=2, lineterm="")
    lines = list(islice(diff, _PREVIEW_LINES + 1))
    if not lines:
        return "未发现正文文本差异。请核对完整文件中的格式、换行或其他内容。", is_text
    rendered = "\n".join(lines[:_PREVIEW_LINES])
    if len(lines) > _PREVIEW_LINES or len(rendered) > _PREVIEW_CHARS:
        rendered = rendered[:_PREVIEW_CHARS] + "\n[预览已截断，完整版本可另行打开]"
    if suffix == ".docx":
        rendered = "DOCX 段落文本预览。排版、图片和批注请在 Word 中比较。\n\n" + rendered
    return rendered, is_text


def _write_review_snapshot(scope: Workspace, relative: Path, data: bytes | None) -> Path | None:
    if data is None:
        return None
    target = scope.resolve(relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if _read_regular(target) != data:
            raise FileTransactionConflict("审阅副本已被编辑。请移走该副本后重新打开对照。")
    except FileNotFoundError:
        _atomic_write_bytes(target, data)
    return target


def compare_workspace_file(
    workspace: Workspace, thread_id: str, user_id: str, relative: str,
    *, expected_token: str | None = None,
) -> FileComparison:
    """Export exact versions outside the container mount."""
    root = _copy_root(workspace, thread_id, user_id)
    if not root.is_dir():
        raise FileNotFoundError("当前会话还没有沙箱副本。")
    with _copy_lock(root):
        review, host, sandbox = _conflict_review(workspace, root, relative, thread_id, user_id)
        token = review["resolution_token"]
        if expected_token is not None and token != expected_token:
            raise FileTransactionConflict("文件版本已变化，请重新打开审阅列表。")
        if host is None and sandbox is None:
            raise FileNotFoundError("原工作区与沙箱均无此文件。")
        name = Path(review["path"]).name
        review_scope = Workspace.from_path(root)
        snapshot = Path("review-snapshots") / token[:24]
        original = _write_review_snapshot(review_scope, snapshot / "original" / name, host)
        candidate = _write_review_snapshot(review_scope, snapshot / "candidate" / name, sandbox)
    preview, is_text = _comparison_preview(host, sandbox, Path(name).suffix.lower())
    return FileComparison(
        review["path"], original, candidate,
        len(host) if host is not None else None,
        len(sandbox) if sandbox is not None else None,
        preview, is_text,
    )
