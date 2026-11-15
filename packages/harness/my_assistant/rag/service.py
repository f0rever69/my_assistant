"""Run verification before releasing any knowledge-base answer."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal

from my_assistant.trace_context import ensure_trace_context

from .evidence import (
    Citation,
    CitationVerifier,
    QuoteRequest,
    Verdict,
    numbers_supported,
)
from .index import KnowledgeIndex
from .model import EvidenceModel

OutputMode = Literal["strict", "annotate"]


@dataclass(frozen=True)
class Finding:
    topic: str
    text: str
    verdict: Verdict
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True)
class Answer:
    trace_id: str
    findings: tuple[Finding, ...]
    error: str | None = None
    mode: OutputMode = "strict"

    @property
    def status(self) -> str:
        verdicts = {item.verdict for item in self.findings}
        if self.error:
            return "unverified"
        if verdicts == {"supported"}:
            return "supported"
        if "supported" in verdicts:
            return "partial"
        if "conflict" in verdicts:
            return "conflict"
        if verdicts == {"contradicted"}:
            return "contradicted"
        return "insufficient"

    def render(self) -> str:
        lines = []
        labels = {
            "supported": "已核验",
            "contradicted": "与资料矛盾",
            "insufficient": "证据不足",
            "conflict": "来源冲突",
        }
        for item in self.findings:
            detail = item.text if item.verdict == "supported" else item.topic
            if self.mode == "annotate" and item.verdict != "supported" and item.text:
                detail += f"（待核实候选：{item.text}）"
            lines.append(f"- {labels[item.verdict]}：{detail}")
            for citation in item.citations:
                hit = citation.hit
                path = _safe_label(hit.path)
                quote = "\n    ".join(
                    _strip_controls(line) for line in citation.quote.split("\n")
                )
                lines.append(
                    f"  - {path}，{hit.location}，版本 {hit.version[:12]}：{quote}"
                )
        if self.error:
            lines.append(f"核验状态：{self.error}")
        return "\n".join(lines or ["证据不足：未找到可核验的资料。"])


def _strip_controls(value: str) -> str:
    return "".join(
        char for char in value if unicodedata.category(char) not in {"Cc", "Cf"}
    )


def _safe_label(value: str) -> str:
    return re.sub(
        r"\s+", " ", _strip_controls(value.replace("\n", " ").replace("\t", " "))
    ).strip()


class KnowledgeService:
    def __init__(self, index: KnowledgeIndex, model: EvidenceModel, *, audit=None):
        self.index = index
        self.model = model
        self.audit = audit

    def ask(
        self,
        question: str,
        *,
        variants: tuple[str, ...] = (),
        mode: OutputMode = "strict",
    ) -> Answer:
        if not question.strip() or len(question) > 500:
            raise ValueError("Question must contain 1–500 characters")
        if mode not in {"strict", "annotate"}:
            raise ValueError("Unknown knowledge base output mode")
        with ensure_trace_context() as trace_id:
            start = perf_counter()
            report = {"trace_id": trace_id, "mode": mode, "stages_ms": {}}
            try:
                answer = self._ask(trace_id, question, variants, report)
                report["error"] = answer.error
                report["status"] = answer.status
                report["verdicts"] = dict(
                    Counter(item.verdict for item in answer.findings)
                )
                report["findings"] = [
                    {"position": position, "verdict": item.verdict}
                    for position, item in enumerate(answer.findings, 1)
                ]
                if mode == "strict":
                    answer = replace(
                        answer,
                        findings=tuple(
                            item
                            if item.verdict == "supported"
                            else replace(item, text="")
                            for item in answer.findings
                        ),
                    )
                return replace(answer, mode=mode)
            except Exception:
                report["error"] = "request_failed"
                raise
            finally:
                report["elapsed_ms"] = round((perf_counter() - start) * 1000)
                if self.audit is not None:
                    self.audit.write(report)

    def _ask(
        self, trace_id: str, question: str, variants: tuple[str, ...], report: dict
    ) -> Answer:
        start = perf_counter()
        revision, hits = self.index.search(question, variants=variants)
        hits = hits[:8]
        report["revision"] = revision
        report["candidates"] = [hit.chunk_id for hit in hits]
        report["stages_ms"]["search"] = round((perf_counter() - start) * 1000)
        if not hits:
            return Answer(trace_id, ())

        start = perf_counter()
        try:
            draft = self.model.extract(
                {
                    "question": question,
                    "passages": [
                        {
                            "chunk_id": hit.chunk_id,
                            "location": hit.location,
                            "text": hit.text,
                        }
                        for hit in hits
                    ],
                }
            )
        except Exception:  # noqa: BLE001 - provider and fixture failures leave the answer unverified
            return Answer(trace_id, (), "extraction_failed")
        finally:
            report["stages_ms"]["extract"] = round((perf_counter() - start) * 1000)

        start = perf_counter()
        verifier = CitationVerifier(self.index, hits)
        findings = []
        evidence_by_claim = {}
        source_ids = {
            path: f"D{number}"
            for number, path in enumerate(dict.fromkeys(hit.path for hit in hits), 1)
        }
        batch = []
        for topic in draft.topics:
            question_label = (
                _safe_label(topic.question)
                if topic.question.strip() in question
                else _safe_label(question)
            )
            if not topic.claims:
                findings.append(Finding(question_label, "", "insufficient"))
            for claim in topic.claims:
                claim_id = str(len(findings) + 1)
                text = _safe_label(claim.text)
                findings.append(Finding(question_label, text, "insufficient"))
                evidence = {}
                for quote in claim.quotes:
                    citation, reason = verifier.verify(quote)
                    report.setdefault("citations", []).append(
                        {
                            "claim_id": claim_id,
                            "status": reason,
                            "chunk_id": quote.chunk_id
                            if quote.chunk_id in verifier.hits
                            else None,
                        }
                    )
                    if citation is not None and citation not in evidence.values():
                        evidence[f"{claim_id}.{len(evidence) + 1}"] = citation
                if evidence and text:
                    evidence_by_claim[claim_id] = evidence
                    batch.append(
                        {
                            "claim_id": claim_id,
                            "claim": text,
                            "evidence": [
                                {
                                    "id": key,
                                    "source": source_ids[item.hit.path],
                                    "location": item.hit.location,
                                    "quote": item.quote,
                                    "context": item.hit.text,
                                }
                                for key, item in evidence.items()
                            ],
                        }
                    )
        report["stages_ms"]["citations"] = round((perf_counter() - start) * 1000)
        if not batch:
            return Answer(trace_id, tuple(findings))
        start = perf_counter()
        try:
            judgments = self.model.verify({"claims": batch}).judgments
        except Exception:  # noqa: BLE001 - provider and fixture failures leave the answer unverified
            return Answer(trace_id, tuple(findings), "verification_failed")
        finally:
            report["stages_ms"]["verify"] = round((perf_counter() - start) * 1000)

        start = perf_counter()
        if self.index.revision() != revision:
            return Answer(trace_id, tuple(findings), "index_changed")
        counts = Counter(judgment.claim_id for judgment in judgments)
        final_check = CitationVerifier(self.index, hits)
        for judgment in judgments:
            claim_id = judgment.claim_id
            if counts[claim_id] != 1 or claim_id not in evidence_by_claim:
                continue
            evidence = evidence_by_claim[claim_id]
            keys = judgment.evidence_ids
            if (
                not keys
                or len(keys) != len(set(keys))
                or not set(keys) <= evidence.keys()
            ):
                continue
            selected = [evidence[key] for key in keys]
            verified = []
            for item in selected:
                citation, reason = final_check.verify(
                    QuoteRequest(chunk_id=item.hit.chunk_id, quote=item.quote)
                )
                report.setdefault("publication_checks", []).append(
                    {
                        "claim_id": claim_id,
                        "chunk_id": item.hit.chunk_id,
                        "status": reason,
                    }
                )
                if citation is not None:
                    verified.append(citation)
            if len(verified) != len(selected):
                continue
            position = int(claim_id) - 1
            finding = findings[position]
            verdict = judgment.verdict
            if verdict == "insufficient":
                continue
            if verdict == "supported" and not numbers_supported(finding.text, verified):
                continue
            if verdict == "conflict" and len({item.hit.path for item in verified}) < 2:
                continue
            findings[position] = replace(
                finding, verdict=verdict, citations=tuple(verified)
            )
        report["stages_ms"]["publish"] = round((perf_counter() - start) * 1000)
        return Answer(trace_id, tuple(findings))
