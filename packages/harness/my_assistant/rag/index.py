"""Private per-workspace BM25 passage index."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from collections import defaultdict
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from my_assistant.config.workspace import Workspace
from my_assistant.sandbox.workspace_search import (
    _SKIPPED_DIRS,
    _quote_fts_term,
    _split_terms,
)

from .documents import (
    EXTRACTOR_VERSION,
    SUFFIXES,
    DocumentError,
    extract_passages,
    read_source,
    source_path,
)

MAX_DOCUMENTS = 500
MAX_ENTRIES = 10_000
MAX_INDEX_CHARS = 10_000_000


def passage_id(relative: str, version: str, ordinal: int) -> str:
    return hashlib.sha256(
        f"{relative}\0{version}\0{EXTRACTOR_VERSION}\0{ordinal}".encode()
    ).hexdigest()


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    path: str
    version: str
    location: str
    text: str


class KnowledgeIndex:
    def __init__(
        self,
        db_path: Path,
        workspace: Workspace,
        *,
        resolve_workspace: Callable[[], Workspace] | None = None,
    ):
        self.db_path = db_path
        self.workspace = workspace
        self.resolve_workspace = resolve_workspace

    def current_workspace(self) -> Workspace:
        workspace = (
            self.resolve_workspace() if self.resolve_workspace else self.workspace
        )
        if (workspace.id, workspace.root_fingerprint) != (
            self.workspace.id,
            self.workspace.root_fingerprint,
        ):
            raise PermissionError("Knowledge base workspace identity changed")
        return workspace

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.db_path, timeout=10)
        try:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS catalog (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS documents (
                    path TEXT PRIMARY KEY, version TEXT NOT NULL, extractor TEXT NOT NULL, chars INTEGER NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5(
                    chunk_id UNINDEXED, path UNINDEXED, location UNINDEXED, text UNINDEXED,
                    title_terms, words, cjk_grams
                );
            """)
        except Exception:
            db.close()
            raise
        return db

    @staticmethod
    def _catalog(db: sqlite3.Connection) -> dict:
        return dict(db.execute("SELECT key, value FROM catalog"))

    def refresh(self, source: str | None = None) -> dict:
        workspace = self.current_workspace()
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            catalog = self._catalog(db)
            if source is None:
                source = catalog.get("source")
                if source is None:
                    raise ValueError("Choose a knowledge base directory using --source")
            root = source_path(workspace, source)
            if not root.is_dir():
                raise ValueError("Knowledge base source must be a directory")
            old = {
                path: (version, extractor, chars)
                for path, version, extractor, chars in db.execute(
                    "SELECT * FROM documents"
                )
            }
            kept = set()
            skipped = []
            entries = imported = total_chars = 0
            for directory, folders, files in os.walk(root, followlinks=False):
                folders[:] = sorted(
                    name
                    for name in folders
                    if not name.startswith(".") and name not in _SKIPPED_DIRS
                )
                for folder in folders[:]:
                    try:
                        workspace.resolve(Path(directory) / folder)
                    except (OSError, PermissionError):
                        folders.remove(folder)
                entries += len(folders) + len(files)
                if entries > MAX_ENTRIES:
                    skipped.append({"path": ".", "reason": "scan_limit"})
                    break
                for name in sorted(files):
                    path = Path(directory) / name
                    if path.suffix.lower() not in SUFFIXES:
                        continue
                    relative = path.relative_to(workspace.root).as_posix()
                    if len(kept) >= MAX_DOCUMENTS:
                        skipped.append({"path": ".", "reason": "document_limit"})
                        break
                    try:
                        raw, version = read_source(workspace, relative)
                        previous = old.get(relative)
                        if previous and previous[:2] == (version, EXTRACTOR_VERSION):
                            chars = previous[2]
                        else:
                            passages = extract_passages(path.suffix.lower(), raw)
                            chars = sum(len(passage.text) for passage in passages)
                        if total_chars + chars > MAX_INDEX_CHARS:
                            raise DocumentError("index_text_limit")
                        if not previous or previous[:2] != (version, EXTRACTOR_VERSION):
                            self._remove(db, relative)
                            title = " ".join(_split_terms(path.stem)[0])
                            for ordinal, passage in enumerate(passages):
                                chunk_id = passage_id(relative, version, ordinal)
                                words, grams = _split_terms(passage.text)
                                db.execute(
                                    "INSERT INTO passages VALUES (?, ?, ?, ?, ?, ?, ?)",
                                    (
                                        chunk_id,
                                        relative,
                                        passage.location,
                                        passage.text,
                                        title,
                                        " ".join(words),
                                        " ".join(grams),
                                    ),
                                )
                            db.execute(
                                "INSERT INTO documents VALUES (?, ?, ?, ?)",
                                (relative, version, EXTRACTOR_VERSION, chars),
                            )
                            imported += 1
                        kept.add(relative)
                        total_chars += chars
                    except (OSError, DocumentError) as exc:
                        reason = (
                            str(exc)
                            if isinstance(exc, DocumentError)
                            else (
                                "not_readable"
                                if not isinstance(exc, PermissionError)
                                else "permission_denied"
                            )
                        )
                        skipped.append(
                            {
                                "path": relative
                                if reason != "permission_denied"
                                else None,
                                "reason": reason,
                            }
                        )
                if len(kept) >= MAX_DOCUMENTS:
                    break
            for relative in old.keys() - kept:
                self._remove(db, relative)
            revision = int(catalog.get("revision", "0")) + 1
            db.executemany(
                "INSERT OR REPLACE INTO catalog VALUES (?, ?)",
                (
                    ("source", root.relative_to(workspace.root).as_posix()),
                    ("revision", str(revision)),
                ),
            )
            return {
                "revision": revision,
                "indexed": len(kept),
                "imported": imported,
                "skipped": skipped,
            }

    @staticmethod
    def _remove(db: sqlite3.Connection, relative: str) -> None:
        db.execute("DELETE FROM passages WHERE path = ?", (relative,))
        db.execute("DELETE FROM documents WHERE path = ?", (relative,))

    def list_documents(self) -> list[dict]:
        workspace = self.current_workspace()
        with closing(self._connect()) as db:
            return [
                {"path": path, "version": version}
                for path, version in db.execute(
                    "SELECT path, version FROM documents ORDER BY path"
                )
                if self._current_version(workspace, path) == version
            ]

    @staticmethod
    def _current_version(workspace: Workspace, path: str) -> str | None:
        try:
            return read_source(workspace, path)[1]
        except (OSError, DocumentError):
            return None

    def revision(self) -> int:
        with closing(self._connect()) as db:
            return int(self._catalog(db).get("revision", "0"))

    def search(
        self, question: str, *, variants: tuple[str, ...] = (), limit: int = 12
    ) -> tuple[int, list[Hit]]:
        if not 1 <= limit <= 40:
            raise ValueError("Search limit must be between 1 and 40")
        workspace = self.current_workspace()
        with closing(self._connect()) as db, db:
            db.execute("BEGIN")
            revision = int(self._catalog(db).get("revision", "0"))
            if revision == 0:
                raise ValueError("Build the knowledge base index first")
            ranks = defaultdict(float)
            hits = {}
            versions = {}
            for query in list(dict.fromkeys((question, *variants)))[:5]:
                words, grams = _split_terms(query[:500])
                terms = list(dict.fromkeys((*words[:20], *grams[:30])))
                if not terms:
                    continue
                match = " OR ".join(map(_quote_fts_term, terms))
                rows = db.execute(
                    """
                    SELECT p.chunk_id, p.path, d.version, p.location, p.text
                    FROM passages p JOIN documents d ON p.path = d.path
                    WHERE passages MATCH ? AND d.extractor = ?
                    ORDER BY bm25(passages, 0, 0, 0, 0, 3, 1, 0.25), p.chunk_id
                    LIMIT ?
                """,
                    (match, EXTRACTOR_VERSION, limit * 4),
                )
                for rank, row in enumerate(rows, 1):
                    hit = Hit(*row)
                    if hit.path not in versions:
                        versions[hit.path] = self._current_version(workspace, hit.path)
                    if versions[hit.path] == hit.version:
                        ranks[hit.chunk_id] += 1 / (60 + rank)
                        hits[hit.chunk_id] = hit
            return revision, [
                hits[key]
                for key in sorted(hits, key=lambda key: (-ranks[key], key))[:limit]
            ]
