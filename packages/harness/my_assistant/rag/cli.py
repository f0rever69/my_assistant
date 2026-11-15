"""Headless knowledge base management and verified question answering."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

from .local import audit_log, open_index, open_service


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="my_assistant kb")
    workspace = parser.add_mutually_exclusive_group(required=True)
    workspace.add_argument("--workspace")
    workspace.add_argument("--workspace-id")
    commands = parser.add_subparsers(dest="command", required=True)
    refresh = commands.add_parser("index", help="build or refresh the index")
    refresh.add_argument("--source", help="workspace-relative knowledge base directory")
    commands.add_parser("list", help="list indexed documents")
    search = commands.add_parser(
        "search", help="return ranked passages without a model"
    )
    search.add_argument("question")
    search.add_argument("--variant", action="append", default=[])
    search.add_argument("--limit", type=int, default=12)
    ask = commands.add_parser("ask", help="answer using verified evidence")
    ask.add_argument("question")
    ask.add_argument("--variant", action="append", default=[])
    ask.add_argument("--mode", choices=("strict", "annotate"))
    ask.add_argument("--mock", type=Path, metavar="FIXTURE.json")
    ask.add_argument(
        "--allow-model-data",
        action="store_true",
        help="allow the configured model to receive selected excerpts",
    )
    ask.add_argument(
        "--json", action="store_true", help="return the structured verified answer"
    )
    audit = commands.add_parser(
        "audit", help="inspect recent metadata-only audit records"
    )
    audit.add_argument("--limit", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        index = open_index(path=args.workspace, workspace_id=args.workspace_id)
        if args.command == "index":
            if args.source is not None and Path(args.source).is_absolute():
                raise ValueError("--source must be relative to the selected Workspace")
            result = index.refresh(args.source)
        elif args.command == "list":
            result = index.list_documents()
        elif args.command == "search":
            revision, hits = index.search(
                args.question, variants=tuple(args.variant), limit=args.limit
            )
            result = {"revision": revision, "hits": [asdict(hit) for hit in hits]}
        elif args.command == "audit":
            from my_assistant.config.rag_config import RagConfig

            result = audit_log(index, RagConfig()).recent(args.limit)
        else:
            service, config = open_service(
                index, mock_path=args.mock, allow_model_data=args.allow_model_data
            )
            answer = service.ask(
                args.question,
                variants=tuple(args.variant),
                mode=args.mode or config.output_mode,
            )
            if args.json:
                result = {
                    "trace_id": answer.trace_id,
                    "mode": answer.mode,
                    "status": answer.status,
                    "error": answer.error,
                    "findings": [
                        {
                            "topic": item.topic,
                            "text": item.text,
                            "verdict": item.verdict,
                            "citations": [
                                {
                                    "path": cite.hit.path,
                                    "location": cite.hit.location,
                                    "version": cite.hit.version,
                                    "chunk_id": cite.hit.chunk_id,
                                    "start": cite.start,
                                    "end": cite.end,
                                    "quote": cite.quote,
                                }
                                for cite in item.citations
                            ],
                        }
                        for item in answer.findings
                    ],
                }
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(answer.render())
                print(f"trace_id: {answer.trace_id}", file=sys.stderr)
            return 1 if answer.error else 0
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (LookupError, OSError, ValueError, sqlite3.Error) as exc:
        print(f"Knowledge base: {exc}", file=sys.stderr)
        return 2
