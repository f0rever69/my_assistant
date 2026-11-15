"""Structured model output and exact evidence checks."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .documents import DocumentError, extract_passages, read_source
from .index import Hit, KnowledgeIndex, passage_id

Verdict = Literal["supported", "contradicted", "insufficient", "conflict"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=False)


class QuoteRequest(StrictModel):
    chunk_id: str = Field(min_length=1, max_length=64)
    quote: str = Field(min_length=1, max_length=650)


class Claim(StrictModel):
    text: str = Field(min_length=1, max_length=400)
    quotes: list[QuoteRequest] = Field(max_length=4)


class Topic(StrictModel):
    question: str = Field(min_length=1, max_length=350)
    claims: list[Claim] = Field(max_length=4)


class Draft(StrictModel):
    topics: list[Topic] = Field(min_length=1, max_length=8)


class Judgment(StrictModel):
    claim_id: str = Field(min_length=1, max_length=16)
    verdict: Verdict
    evidence_ids: list[str] = Field(max_length=4)


class Judgments(StrictModel):
    judgments: list[Judgment] = Field(max_length=32)


@dataclass(frozen=True)
class Citation:
    hit: Hit
    start: int
    end: int

    @property
    def quote(self) -> str:
        return self.hit.text[self.start : self.end]


class CitationVerifier:
    def __init__(self, index: KnowledgeIndex, hits: list[Hit]):
        self.index = index
        self.hits = {hit.chunk_id: hit for hit in hits}
        self.sources = {}

    def verify(self, request: QuoteRequest) -> tuple[Citation | None, str]:
        hit = self.hits.get(request.chunk_id)
        if hit is None:
            return None, "unknown_chunk"
        if hit.path not in self.sources:
            try:
                raw, version = read_source(self.index.current_workspace(), hit.path)
                if version != hit.version:
                    return None, "stale_source"
                self.sources[hit.path] = (
                    version,
                    {
                        passage_id(hit.path, version, ordinal): passage
                        for ordinal, passage in enumerate(
                            extract_passages(Path(hit.path).suffix.lower(), raw)
                        )
                    },
                )
            except (OSError, DocumentError, LookupError):
                return None, "source_unavailable"
        version, passages = self.sources[hit.path]
        passage = passages.get(hit.chunk_id)
        if (
            version != hit.version
            or passage is None
            or (passage.text, passage.location) != (hit.text, hit.location)
        ):
            return None, "source_mismatch"
        start = hit.text.find(request.quote) if request.quote.strip() else -1
        if start < 0:
            return None, "quote_mismatch"
        if hit.text.find(request.quote, start + 1) >= 0:
            return None, "ambiguous_quote"
        return Citation(hit, start, start + len(request.quote)), "verified"


_NUMBER = re.compile(r"(?<![0-9])[-+]?\d+(?:[.,]\d+)*(?:\s*[%％])?")
_QUANTITY = re.compile(
    rf"({_NUMBER.pattern})\s*(工作日|自然日|个?小时|分钟|秒|天|日|年|月|万元|亿元|元|公里|千克|公斤|米|次|张|份)"
)
_UNIT_ALIASES = {"日": "天", "个小时": "小时", "公斤": "千克"}


def numbers_supported(claim: str, citations: list[Citation]) -> bool:
    claim = unicodedata.normalize("NFKC", claim)
    quotes = [unicodedata.normalize("NFKC", item.quote) for item in citations]

    def numbers(text: str) -> set[str]:
        return {match.replace(" ", "") for match in _NUMBER.findall(text)}

    def quantities(text: str) -> set[tuple[str, str]]:
        return {
            (number.replace(" ", ""), _UNIT_ALIASES.get(unit, unit))
            for number, unit in _QUANTITY.findall(text)
        }

    return numbers(claim) <= set().union(
        *(numbers(quote) for quote in quotes)
    ) and quantities(claim) <= set().union(*(quantities(quote) for quote in quotes))
