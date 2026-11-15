"""Small deterministic policy checks shared by memory write and read paths."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from typing import Any

_SENSITIVE_PATTERNS = (
    re.compile(
        r"\b(password|passwd|passphrase|secret|api[_ -]?key|access[_ -]?token|private[_ -]?key)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(ssn|social security|credit card|card number|bank account|routing number)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(diagnos(ed|is)|medical|prescription|health condition|health record|" 
        r"mental health|disorder|disease|exact home address|home address|phone number)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b"
    ),
)


def is_sensitive_memory_text(value: str) -> bool:
    """Return True for content that should not enter durable memory by default."""
    return any(pattern.search(value) for pattern in _SENSITIVE_PATTERNS)


def fact_expired(fact: dict[str, Any], *, now: datetime | None = None) -> bool:
    """Evaluate the optional per-fact TTL from its last confirmation or creation."""
    raw_ttl = fact.get("ttlDays")
    if isinstance(raw_ttl, bool) or not isinstance(raw_ttl, (int, float)) or raw_ttl <= 0:
        return False
    timestamp = fact.get("lastConfirmedAt") or fact.get("createdAt")
    if not isinstance(timestamp, str):
        return False
    try:
        created = datetime.fromisoformat(timestamp)
    except ValueError:
        return False
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return (now or datetime.now(UTC)) >= created + timedelta(days=int(raw_ttl))


def strip_sensitive_summaries(memory: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow document copy with sensitive summary text removed."""
    result = dict(memory)
    for section_name in ("user", "history"):
        section = memory.get(section_name)
        if not isinstance(section, dict):
            continue
        cleaned = dict(section)
        for key, value in cleaned.items():
            if isinstance(value, dict) and isinstance(value.get("summary"), str):
                cleaned[key] = dict(value)
                if is_sensitive_memory_text(value["summary"]):
                    cleaned[key]["summary"] = ""
        result[section_name] = cleaned
    return result
