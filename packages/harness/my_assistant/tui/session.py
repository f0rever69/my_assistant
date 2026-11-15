"""Embedded session wiring for the TUI.

Owns construction of the ``MyAssistantClient`` (with a persistent checkpointer),
thread resolution for ``--continue`` / ``--resume`` (by id **or** title), and local
session metadata persistence (see ``my_assistant.tui.persistence``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid importing the heavy client during pure planning
    from my_assistant.client import MyAssistantClient

    from .cli import LaunchPlan
    from .persistence import ThreadMetaWriter, _LoopThread
    from .view_state import ViewState


@dataclass
class Session:
    client: MyAssistantClient
    writer: ThreadMetaWriter | None = None
    _loop: _LoopThread | None = None
    _owns_checkpointer: bool = False

    def resolve_thread(self, plan: LaunchPlan) -> str | None:
        """Resolve the thread id to run against, honoring --resume / --continue."""
        if plan.thread_id:
            return self.resolve_ref(plan.thread_id)
        if plan.continue_recent:
            threads = self.recent_threads(limit=1)
            if threads:
                return threads[0].get("thread_id")
        return None

    def resolve_ref(self, ref: str) -> str:
        """Resolve a thread reference (id or title) to a thread id.

        Matches an existing thread by id first, then by exact title. Falls back to
        the literal ref (treated as an id) when nothing matches, so an unknown id
        still continues/creates that namespace — provided it satisfies the
        canonical thread ID contract.
        """
        try:
            threads = self.recent_threads(limit=100, include_archived=True)
        except Exception:  # noqa: BLE001 - resolution is best-effort
            return self._validated_literal_ref(ref)
        if any(t.get("thread_id") == ref for t in threads):
            return ref
        for thread in threads:
            if (thread.get("title") or "") == ref:
                return thread.get("thread_id") or self._validated_literal_ref(ref)
        return self._validated_literal_ref(ref)

    @staticmethod
    def _validated_literal_ref(ref: str) -> str:
        """Validate a literal ref before it is adopted as a thread id."""
        from my_assistant.utils.thread_id import validate_thread_id

        try:
            return validate_thread_id(ref)
        except ValueError as exc:
            raise ValueError(f"Thread reference {ref!r} matches no existing thread and is not a valid thread id (expected 1-64 ASCII letters, digits, hyphens, or underscores).") from exc

    def recent_threads(self, limit: int = 20, *, include_archived: bool = False, archived_only: bool = False) -> list[dict]:
        """Return a metadata-enriched, recency-sorted local thread catalog.

        Checkpoints remain the source of conversation content. ``threads_meta``
        contributes user-facing lifecycle fields (manual title and archive
        status). Legacy checkpoint-only threads are kept visible.
        """
        checkpoint_limit = max(200, limit * 20)
        checkpoint_threads = self.client.list_threads(limit=checkpoint_limit).get("thread_list", [])
        merged: dict[str, dict] = {
            row["thread_id"]: dict(row)
            for row in checkpoint_threads
            if isinstance(row, dict) and row.get("thread_id")
        }

        writer = self.writer
        metadata_rows = writer.search(archived=None, limit=max(200, limit * 4)) if writer is not None else []
        for meta in metadata_rows:
            thread_id = meta.get("thread_id")
            if not thread_id:
                continue
            row = merged.setdefault(
                thread_id,
                {
                    "thread_id": thread_id,
                    "created_at": meta.get("created_at"),
                    "updated_at": meta.get("updated_at"),
                    "latest_checkpoint_id": None,
                    "title": None,
                },
            )
            metadata = dict(meta.get("metadata") or {})
            display_name = meta.get("display_name")
            if display_name:
                row["title"] = display_name
            row["created_at"] = row.get("created_at") or meta.get("created_at")
            if str(meta.get("updated_at") or "") > str(row.get("updated_at") or ""):
                row["updated_at"] = meta.get("updated_at")
            row["metadata"] = metadata
            row["archived"] = metadata.get("my_assistant_archived") is True
            row["manual_title"] = metadata.get("my_assistant_tui_manual_title") is True

        rows = list(merged.values())
        for row in rows:
            row.setdefault("metadata", {})
            row.setdefault("archived", False)
            row.setdefault("manual_title", False)

        if archived_only:
            rows = [row for row in rows if row["archived"]]
        elif not include_archived:
            rows = [row for row in rows if not row["archived"]]
        rows.sort(key=lambda row: (str(row.get("updated_at") or row.get("created_at") or ""), str(row.get("thread_id") or "")), reverse=True)
        return rows[:limit]

    def get_thread_info(self, thread_id: str) -> dict | None:
        for thread in self.recent_threads(limit=500, include_archived=True):
            if thread.get("thread_id") == thread_id:
                return thread
        return None

    def load_view_state(self, thread_id: str) -> ViewState:
        """Materialize the latest checkpoint into TUI rows."""
        from .runtime import restore_view_state

        payload = self.client.get_thread(thread_id)
        info = self.get_thread_info(thread_id) or {}
        return restore_view_state(payload, title_override=info.get("title"))

    def rename_thread(self, thread_id: str, title: str) -> bool:
        title = title.strip()
        if not title:
            raise ValueError("Thread title must not be empty.")
        if len(title) > 256:
            raise ValueError("Thread title must be 256 characters or fewer.")
        if self.writer is None:
            return False
        return self.writer.set_title(thread_id, title, manual=True)

    def archive_thread(self, thread_id: str, *, archived: bool = True) -> bool:
        if self.writer is None:
            return False
        return self.writer.set_archived(thread_id, archived)

    def delete_thread(self, thread_id: str) -> None:
        """Delete checkpoint state, local thread files, and catalog metadata."""
        from my_assistant.config.paths import get_paths
        from my_assistant.runtime.user_context import DEFAULT_USER_ID
        from my_assistant.utils.thread_id import validate_thread_id

        validate_thread_id(thread_id)
        checkpointer = self.client._get_thread_checkpointer()
        delete = getattr(checkpointer, "delete_thread", None)
        if delete is None:
            raise RuntimeError(f"{type(checkpointer).__name__} does not support thread deletion")
        delete(thread_id)
        get_paths().delete_thread_dir(thread_id, user_id=DEFAULT_USER_ID)
        if self.writer is not None:
            self.writer.delete(thread_id)

    def close(self) -> None:
        """Stop the background DB loop and dispose the engine (best-effort)."""
        if self._owns_checkpointer:
            self._owns_checkpointer = False
            try:
                from my_assistant.runtime.checkpointer.provider import (
                    reset_checkpointer,
                )

                reset_checkpointer()
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
        loop = self._loop
        if loop is None:
            return
        self._loop = None
        try:
            from my_assistant.persistence.engine import close_engine

            loop.run(close_engine())
        except Exception:  # noqa: BLE001 - teardown is best-effort
            pass
        loop.close()


def open_session(persistence: bool = True) -> Session:
    """Build an embedded session backed by the configured checkpointer.

    ``persistence`` controls the shared ``threads_meta`` writer (and its background
    DB loop/engine). Headless one-shots never use the writer, so they pass
    ``persistence=False`` to avoid standing up an event loop + connection pool only
    to discard it.
    """
    from my_assistant.client import MyAssistantClient
    from my_assistant.runtime.checkpointer.provider import get_checkpointer

    checkpointer = get_checkpointer()
    if not persistence:
        client = MyAssistantClient(checkpointer=checkpointer)
        return Session(client=client, _owns_checkpointer=True)

    from .persistence import build_persistence

    loop, writer = build_persistence()
    client = MyAssistantClient(checkpointer=checkpointer, async_runner=loop.run)
    return Session(client=client, writer=writer, _loop=loop, _owns_checkpointer=True)
