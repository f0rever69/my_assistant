"""Read bounded, authorized source bytes and extract text with source locations."""

from __future__ import annotations

import hashlib
import io
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from my_assistant.config.workspace import Workspace
from my_assistant.workspace_changes.scanner import is_sensitive_workspace_path

SUFFIXES = {".txt", ".md", ".pdf", ".docx"}
EXTRACTOR_VERSION = "1"
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 250_000
MAX_CHUNK_CHARS = 750
MAX_PAGES = 100
_WORD_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class DocumentError(ValueError):
    """A source cannot be used as evidence."""


@dataclass(frozen=True)
class Passage:
    location: str
    text: str


def source_path(workspace: Workspace, relative: str | Path) -> Path:
    path = workspace.check(relative)
    scoped = path.relative_to(workspace.root)
    if any(
        part.startswith(".") for part in scoped.parts
    ) or is_sensitive_workspace_path(scoped.as_posix()):
        raise PermissionError("Private path is excluded from the knowledge base")
    return path


def read_source(workspace: Workspace, relative: str | Path) -> tuple[bytes, str]:
    path = source_path(workspace, relative)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise DocumentError("not_a_regular_file")
    if before.st_size > MAX_FILE_BYTES:
        raise DocumentError("file_too_large")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    with os.fdopen(os.open(path, flags), "rb") as stream:
        if not _same_file(before, os.fstat(stream.fileno())):
            raise DocumentError("source_changed")
        raw = stream.read(MAX_FILE_BYTES + 1)
        if not _same_file(before, os.fstat(stream.fileno())):
            raise DocumentError("source_changed")
    if len(raw) != before.st_size or not _same_file(
        before, source_path(workspace, relative).lstat()
    ):
        raise DocumentError("source_changed")
    return raw, hashlib.sha256(raw).hexdigest()


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino, left.st_size, left.st_mtime_ns) == (
        right.st_dev,
        right.st_ino,
        right.st_size,
        right.st_mtime_ns,
    )


def _text_paragraphs(text: str) -> list[Passage]:
    passages = []
    lines = []
    start = 1
    for number, line in enumerate([*text.split("\n"), ""], 1):
        if lines and (
            not line.strip()
            or line.startswith("#")
            or sum(map(len, lines)) >= MAX_CHUNK_CHARS
        ):
            passages.append(Passage(f"行 {start}–{number - 1}", "\n".join(lines)))
            lines = []
        if line.strip():
            if not lines:
                start = number
            lines.append(line)
    return passages


def _pdf_pages(raw: bytes) -> list[Passage]:
    # The existing PDF conversion dependencies include pdfminer. Its page API
    # preserves the page boundary even when a preceding page has no text.
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer

    passages = []
    for page_number, page in enumerate(
        extract_pages(io.BytesIO(raw), maxpages=MAX_PAGES + 1), 1
    ):
        if page_number > MAX_PAGES:
            raise DocumentError("too_many_pdf_pages")
        passages.extend(
            Passage(f"第 {page_number} 页", item.get_text())
            for item in page
            if isinstance(item, LTTextContainer)
        )
        if sum(len(item.text) for item in passages) > MAX_TEXT_CHARS:
            raise DocumentError("extracted_text_too_large")
    return passages


def _docx_paragraphs(raw: bytes) -> list[Passage]:
    from defusedxml.ElementTree import fromstring

    with ZipFile(io.BytesIO(raw)) as archive:
        member = archive.getinfo("word/document.xml")
        if member.file_size > MAX_FILE_BYTES:
            raise DocumentError("docx_xml_too_large")
        root = fromstring(archive.read(member))
    return [
        Passage(
            f"第 {number} 段",
            "".join(node.text or "" for node in paragraph.iter(f"{_WORD_NS}t")),
        )
        for number, paragraph in enumerate(root.iter(f"{_WORD_NS}p"), 1)
    ]


def extract_passages(suffix: str, raw: bytes) -> list[Passage]:
    try:
        if suffix in {".md", ".txt"}:
            text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
            if "\x00" in text:
                raise DocumentError("not_text")
            if len(text) > MAX_TEXT_CHARS:
                raise DocumentError("extracted_text_too_large")
            passages = _text_paragraphs(text)
        elif suffix == ".pdf":
            passages = _pdf_pages(raw)
        elif suffix == ".docx":
            passages = _docx_paragraphs(raw)
        else:
            raise DocumentError("unsupported_format")
    except DocumentError:
        raise
    except (UnicodeError, BadZipFile, KeyError) as exc:
        raise DocumentError("invalid_document") from exc
    except Exception as exc:
        raise DocumentError("extraction_failed") from exc

    if sum(len(item.text) for item in passages) > MAX_TEXT_CHARS:
        raise DocumentError("extracted_text_too_large")
    chunks = [
        Passage(item.location, item.text[start : start + MAX_CHUNK_CHARS])
        for item in passages
        for start in range(0, len(item.text), MAX_CHUNK_CHARS)
        if item.text[start : start + MAX_CHUNK_CHARS].strip()
    ]
    if not chunks:
        raise DocumentError("no_extractable_text")
    return chunks
