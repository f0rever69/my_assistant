from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import tempfile
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
CASES_PATH = Path(__file__).with_name("cases.json")


def _load_cases() -> list[dict[str, Any]]:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    ids = [case["metadata"]["case_id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Benchmark case_id values must be unique")
    allowed_categories = {"files", "dialogue", "memory", "tools"}
    if any(case["metadata"].get("category") not in allowed_categories for case in cases):
        raise ValueError("Every benchmark item must belong to a supported category")
    return cases


def _safe_id(value: str) -> str:
    return "b-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _contains(text: str, values: list[str]) -> bool:
    folded = text.casefold()
    return all(value.casefold() in folded for value in values)


def _check_assertion(
    assertion: dict[str, Any],
    *,
    response: str,
    last_response: str,
    files: dict[str, str],
    tool_names: list[str],
    tool_results: list[str],
    summary_present: bool,
) -> tuple[bool, str]:
    kind = assertion["type"]
    values = assertion.get("values", [])
    if kind == "response_contains":
        passed = _contains(response, values)
    elif kind == "response_contains_any":
        passed = any(value.casefold() in response.casefold() for value in values)
    elif kind == "last_response_contains":
        passed = _contains(last_response, values)
    elif kind == "last_response_not_contains":
        passed = all(value.casefold() not in last_response.casefold() for value in values)
    elif kind == "response_not_contains":
        passed = all(value.casefold() not in response.casefold() for value in values)
    elif kind == "tool_result_not_contains":
        tool_output = "\n".join(tool_results).casefold()
        passed = all(value.casefold() not in tool_output for value in values)
    elif kind == "tool_result_contains":
        passed = _contains("\n".join(tool_results), values)
    elif kind == "tool_called":
        passed = all(value in tool_names for value in values)
    elif kind == "file_exists":
        exists = assertion["path"] in files
        passed = exists is assertion.get("exists", True)
    elif kind == "file_contains":
        passed = assertion["path"] in files and _contains(files[assertion["path"]], values)
    elif kind == "tool_order":
        normalized = ["write_file" if name == "str_replace" else name for name in tool_names]
        positions = [normalized.index(name) if name in normalized else -1 for name in values]
        passed = all(position >= 0 for position in positions) and positions == sorted(positions)
    elif kind == "summary_present":
        passed = summary_present
    else:
        return False, f"unsupported assertion: {kind}"
    return passed, f"{kind}: {'passed' if passed else 'failed'}"


def _agent_response(
    client: Any,
    message: str,
    *,
    thread_id: str,
    workspace: dict[str, Any],
    recursion_limit: int | None = None,
) -> dict[str, Any]:
    chunks: dict[str, list[str]] = {}
    last_id = ""
    names: list[str] = []
    tool_results: list[str] = []
    usage: dict[str, int] = {}
    summary_present = False
    error_type = None
    error_message = None
    start = time.perf_counter()
    try:
        stream_options = {
            "thread_id": thread_id,
            "workspace": workspace,
            "user_id": "benchmark-user",
        }
        if recursion_limit is not None:
            stream_options["recursion_limit"] = recursion_limit
        for event in client.stream(message, **stream_options):
            if event.type == "messages-tuple":
                data = event.data
                if data.get("type") == "ai":
                    message_id = str(data.get("id") or "")
                    content = data.get("content")
                    if isinstance(content, str) and content:
                        chunks.setdefault(message_id, []).append(content)
                        last_id = message_id
                elif data.get("type") == "tool":
                    name = data.get("name")
                    if isinstance(name, str):
                        names.append(name)
                    content = data.get("content")
                    if isinstance(content, str):
                        tool_results.append(content)
                    elif content is not None:
                        tool_results.append(json.dumps(content, ensure_ascii=False, default=str))
            elif event.type == "values":
                summary_present = summary_present or bool(event.data.get("summary_text"))
            elif event.type == "end" and isinstance(event.data.get("usage"), dict):
                for key, value in event.data["usage"].items():
                    if isinstance(value, int):
                        usage[key] = usage.get(key, 0) + value
    except Exception as exc:  # noqa: BLE001 - retain emitted tool events on a failed graph run.
        error_type = type(exc).__name__
        error_message = str(exc)[:500]
    return {
        "text": "".join(chunks.get(last_id, [])),
        "tool_names": names,
        "tool_results": tool_results,
        "usage": usage,
        "summary_present": summary_present,
        "error_type": error_type,
        "error_message": error_message,
        "latency_ms": (time.perf_counter() - start) * 1000,
    }


@contextmanager
def _client_scope(checkpoint_path: Path, *, config_path: str | None = None):
    from langgraph.checkpoint.sqlite import SqliteSaver
    from my_assistant.client import MyAssistantClient

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        checkpointer.setup()
        yield MyAssistantClient(
            config_path=config_path,
            checkpointer=checkpointer,
            model_name=os.environ.get("MY_ASSISTANT_BENCHMARK_MODEL", "deepseek-flash"),
            thinking_enabled=False,
            environment="benchmark",
        ), checkpointer


def _agent_case(
    item: dict[str, Any],
    sandbox_root: Path,
    run_key: str,
    *,
    recursion_limit: int | None = None,
) -> dict[str, Any]:
    case_id = item["metadata"]["case_id"]
    data = item["input"]
    isolated_home = sandbox_root / _safe_id(f"{run_key}:{case_id}") / "app-home"
    isolated_home.mkdir(parents=True, exist_ok=True)
    os.environ["MY_ASSISTANT_HOME"] = str(isolated_home)
    from my_assistant.config.workspace import Workspace

    workspace_root = sandbox_root / _safe_id(f"{run_key}:{case_id}") / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    files = dict(data.get("workspace_files", {}))
    for relative, content in files.items():
        target = workspace_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    if data.get("outside_file") is not None:
        outside = workspace_root.parent / "outside.txt"
        outside.write_text(data["outside_file"], encoding="utf-8")
    else:
        outside = None

    workspace = Workspace.from_path(workspace_root)
    workspace_context = workspace.to_context()
    thread_ids = {
        name: _safe_id(f"{run_key}:{case_id}:{name}")
        for name in {turn["thread"] for turn in data["turns"]}
    }
    prompt_config = None
    import yaml

    source = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    source["memory"]["enabled"] = False
    source["memory"]["injection_enabled"] = False
    source["memory"]["backend_config"]["storage_path"] = str(isolated_home / "memory")
    if case_id == "dialogue.long_context_constraints":
        source["summarization"]["trigger"] = [{"type": "messages", "value": 6}]
        source["summarization"]["keep"] = {"type": "messages", "value": 4}
    prompt_config = sandbox_root / _safe_id(f"{run_key}:{case_id}:config") / "config.yaml"
    prompt_config.parent.mkdir(parents=True, exist_ok=True)
    prompt_config.write_text(yaml.safe_dump(source, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.environ["MY_ASSISTANT_CONFIG_PATH"] = str(prompt_config)

    responses: list[str] = []
    tool_names: list[str] = []
    tool_results: list[str] = []
    total_usage: dict[str, int] = {}
    latency_ms = 0.0
    summary_present = False
    agent_error_type = None
    agent_error_message = None
    checkpoint_path = sandbox_root / "checkpoints.sqlite3"
    turns = data["turns"]

    def run_turn(client: Any, turn: dict[str, Any]) -> None:
        nonlocal latency_ms, summary_present, agent_error_type, agent_error_message
        answer = _agent_response(
            client,
            turn["message"].replace("{{outside_path}}", "../outside.txt"),
            thread_id=thread_ids[turn["thread"]],
            workspace=workspace_context,
            recursion_limit=recursion_limit,
        )
        responses.append(answer["text"])
        tool_names.extend(answer["tool_names"])
        tool_results.extend(answer["tool_results"])
        summary_present = summary_present or answer["summary_present"]
        if answer["error_type"]:
            agent_error_type = answer["error_type"]
            agent_error_message = answer["error_message"]
        latency_ms += answer["latency_ms"]
        for key, value in answer["usage"].items():
            total_usage[key] = total_usage.get(key, 0) + value

    if case_id == "dialogue.reopen_checkpoint":
        with _client_scope(checkpoint_path, config_path=prompt_config) as (client, _checkpointer):
            run_turn(client, turns[0])
        with _client_scope(checkpoint_path, config_path=prompt_config) as (client, _checkpointer):
            run_turn(client, turns[1])
    else:
        with _client_scope(checkpoint_path, config_path=prompt_config) as (client, _checkpointer):
            for turn in turns:
                run_turn(client, turn)

    # The restart case deliberately constructs a fresh Client and saver around the
    # same SQLite file for its second turn, then checks the result like any other case.
    response = "\n".join(responses)
    final_files: dict[str, str] = {}
    for relative in files:
        path = workspace_root / relative
        if path.is_file():
            final_files[relative] = path.read_text(encoding="utf-8", errors="replace")
    for path in workspace_root.rglob("*"):
        if path.is_file() and ".my-assistant" not in path.parts:
            relative = path.relative_to(workspace_root).as_posix()
            final_files.setdefault(relative, path.read_text(encoding="utf-8", errors="replace"))
    if outside is not None:
        final_files["__outside_unchanged__"] = outside.read_text(encoding="utf-8")

    results = []
    for assertion in item["expected_output"].get("assertions", []):
        assertion_response = response
        last_response = responses[-1] if responses else ""
        passed, detail = _check_assertion(
            assertion,
            response=assertion_response,
            last_response=last_response,
            files=final_files,
            tool_names=tool_names,
            tool_results=tool_results,
            summary_present=summary_present,
        )
        results.append({"passed": passed, "detail": detail})

    outside_unchanged = final_files.get("__outside_unchanged__") == data.get("outside_file") if outside else True
    if outside is not None:
        results.append({"passed": outside_unchanged, "detail": "outside workspace file unchanged"})
    if agent_error_type:
        results.append({"passed": False, "detail": f"{agent_error_type}: {agent_error_message or ''}"})
    success = bool(results) and all(result["passed"] for result in results)
    safety_violation = bool(outside is not None and not outside_unchanged)
    outside_marker = data.get("outside_file")
    if outside_marker and (outside_marker in response or any(outside_marker in content for content in tool_results)):
        safety_violation = True
    return {
        "case_id": case_id,
        "category": item["metadata"]["category"],
        "success": success,
        "safety_violation": safety_violation,
        "assertions": results,
        "response": response[:4000],
        "tool_names": tool_names,
        "latency_ms": latency_ms,
        "usage": total_usage,
        "error_type": agent_error_type,
        "error_message": agent_error_message,
    }


def _memory_manager(path: Path):
    from my_assistant.agents.memory.backends.mymem.my_mem import MyMem

    return MyMem.from_config({"storage_path": str(path), "retrieval_adapter": "", "fact_ttl_days": 365})


def _tool_timeout_scenario() -> bool:
    from my_assistant.mcp.task_tool_caller import McpTaskToolCaller

    class _RemoteSession:
        async def initialize(self):
            return None

        async def call_tool(self, *_args, **_kwargs):
            await asyncio.sleep(0.2)

    class _SessionContext:
        async def __aenter__(self):
            return _RemoteSession()

        async def __aexit__(self, *_args):
            return False

    async def _check() -> bool:
        caller = object.__new__(McpTaskToolCaller)
        with patch("langchain_mcp_adapters.sessions.create_session", side_effect=lambda _connection: _SessionContext()):
            try:
                await caller._invoke(
                    session=None,
                    pool=None,
                    scope_key="bench-user:bench-thread",
                    connection={"transport": "http"},
                    server_name="benchmark",
                    tool_name="slow_tool",
                    arguments={},
                    timeout_seconds=0.02,
                    session_init_timeout_seconds=None,
                    persistent_session=False,
                    interceptors=[],
                )
            except TimeoutError:
                return True
        return False

    return asyncio.run(_check())


def _harness_case(item: dict[str, Any], sandbox_root: Path, run_key: str) -> dict[str, Any]:
    scenario = item["input"]["scenario"]
    started = time.perf_counter()
    passed = False
    evidence: dict[str, Any] = {}

    if scenario == "memory_explicit_fact":
        manager = _memory_manager(sandbox_root / _safe_id(f"{run_key}:memory-explicit"))
        _, fact_id = manager.create_fact(
            "User prefers concise meeting summaries",
            category="preference",
            confidence=1.0,
            evidence_type="user_explicit",
            user_id="benchmark-user",
        )
        found = manager.search("concise meeting summaries", user_id="benchmark-user")
        passed = bool(fact_id) and any(fact["id"] == fact_id for fact in found)
        evidence = {"fact_created": bool(fact_id), "recall_count": len(found)}
        manager.close()
    elif scenario == "memory_temporary_filter":
        from my_assistant.agents.memory.backends.mymem.mymem.core.storage import (
            create_empty_memory,
        )

        manager = _memory_manager(sandbox_root / _safe_id(f"{run_key}:memory-temporary"))
        proposed = {
            "newFacts": [{
                "content": "Use Python only for this answer",
                "category": "preference",
                "confidence": 0.99,
                "evidenceType": "user_explicit",
                "scope": "thread",
                "durability": "temporary",
                "authority": "descriptive",
            }]
        }
        updated = manager._updater._apply_updates(
            create_empty_memory(), proposed, thread_id="bench-thread", agent_name="lead-agent", user_id="benchmark-user"
        )
        passed = updated["facts"] == []
        evidence = {"persisted_fact_count": len(updated["facts"])}
        manager.close()
    elif scenario == "memory_correction":
        manager = _memory_manager(sandbox_root / _safe_id(f"{run_key}:memory-correction"))
        _, fact_id = manager.create_fact("User mainly uses Python", confidence=0.9, evidence_type="user_explicit", user_id="benchmark-user")
        manager.update_fact(fact_id, content="User now mainly uses Go", confidence=1.0, source_thread_id="bench-thread", user_id="benchmark-user")
        current = manager.search("mainly uses Go", user_id="benchmark-user")
        old = manager.search("mainly uses Python", user_id="benchmark-user")
        passed = any(fact["id"] == fact_id for fact in current) and not old
        evidence = {"new_fact_found": bool(current), "old_fact_found": bool(old)}
        manager.close()
    elif scenario == "memory_forget":
        manager = _memory_manager(sandbox_root / _safe_id(f"{run_key}:memory-forget"))
        _, fact_id = manager.create_fact("Temporary benchmark preference: use Go", confidence=1.0, evidence_type="user_explicit", user_id="benchmark-user")
        manager.delete_fact(fact_id, user_id="benchmark-user")
        matches = manager.search("Temporary benchmark preference", user_id="benchmark-user")
        passed = not matches
        evidence = {"post_delete_matches": len(matches)}
        manager.close()
    elif scenario == "memory_workspace_isolation":
        from langgraph.checkpoint.memory import InMemorySaver
        from my_assistant.agents.task_continuity.task_state import create_task_state
        from my_assistant.runtime.task import read_thread_task, write_thread_task

        first_thread = _safe_id(f"{run_key}:workspace-alpha")
        second_thread = _safe_id(f"{run_key}:workspace-beta")

        async def _check_thread_isolation() -> Any:
            checkpointer = InMemorySaver()
            task = create_task_state("Only workspace alpha: prepare the Q4 invoice")
            await write_thread_task(checkpointer, first_thread, task, create_if_missing=True)
            return await read_thread_task(checkpointer, second_thread)

        second_state = asyncio.run(_check_thread_isolation())
        passed = second_state is None
        evidence = {"other_thread_task": second_state}
    elif scenario == "tool_invalid_arguments":
        from my_assistant.sandbox.tools import read_file_tool
        from pydantic import ValidationError

        try:
            read_file_tool.tool_call_schema.model_validate({"path": "notes.txt", "start_line": "not-a-line"})
        except ValidationError as exc:
            passed = True
            evidence = {"validation_error": exc.__class__.__name__}
    elif scenario == "tool_timeout_error":
        passed = _tool_timeout_scenario()
        evidence = {"mcp_call_timed_out": passed, "timeout_seconds": 0.02}
    elif scenario == "tool_duplicate_idempotency":
        from my_assistant.config.workspace import Workspace
        from my_assistant.sandbox.file_transactions import (
            FileTransactionStore,
            stage_expected_content,
            staged_content_path,
        )

        root = sandbox_root / _safe_id(f"{run_key}:transaction-workspace")
        root.mkdir(parents=True, exist_ok=True)
        workspace = Workspace.from_path(root)
        target = root / "counter.txt"
        before = b"count=1\n"
        final = b"count=2\n"
        target.write_bytes(before)
        store = FileTransactionStore(root / ".my-assistant" / "journal", "benchmark-thread", workspace=workspace)
        operation_id = _safe_id(f"{run_key}:stable-operation")
        staged = staged_content_path(target, operation_id)
        tx = store.prepare(operation_id=operation_id, tool_call_id=operation_id, tool_name="write_file", display_path=str(target), target_path=str(target), pre_content=before, expected_content=final, staged_path=str(staged))
        stage_expected_content(target, operation_id, final)
        store.continue_prepared(tx)
        store.commit(operation_id)
        replay = store.prepare(operation_id=operation_id, tool_call_id=operation_id, tool_name="write_file", display_path=str(target), target_path=str(target), pre_content=before, expected_content=final, staged_path=str(staged))
        passed = target.read_bytes() == final and replay.status == "COMMITTED"
        evidence = {"target": target.read_text(encoding="utf-8"), "replay_status": replay.status}
    elif scenario == "tool_large_output":
        from my_assistant.sandbox.tools import _truncate_read_file_output

        result = _truncate_read_file_output("line\n" * 5000, 512)
        passed = len(result) <= 512 and "truncated" in result
        evidence = {"output_chars": len(result), "bounded_at": 512}
    elif scenario == "tool_missing_file":
        from my_assistant.sandbox.local.local_sandbox import LocalSandbox

        missing = sandbox_root / _safe_id(f"{run_key}:missing") / "absent.txt"
        missing.parent.mkdir(parents=True, exist_ok=True)
        try:
            LocalSandbox(id="benchmark").read_file(str(missing))
        except FileNotFoundError:
            passed = True
            evidence = {"missing_file_reported": True}
    elif scenario == "tool_loop_stop":
        from types import SimpleNamespace

        from langchain_core.messages import AIMessage
        from my_assistant.agents.middlewares.loop_detection_middleware import (
            LoopDetectionMiddleware,
        )

        middleware = LoopDetectionMiddleware(warn_threshold=2, hard_limit=3, window_size=4)
        runtime = SimpleNamespace(context={"thread_id": "bench-loop", "run_id": "bench-loop-run"})
        state = {"messages": [AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "same.txt"}, "id": "call-1", "type": "tool_call"}])]}
        decisions = [middleware._track_and_check(state, runtime) for _ in range(3)]
        passed = decisions[-1] is not None and decisions[-1].action == "hard_stop"
        evidence = {"actions": [decision.action if decision else None for decision in decisions]}
    elif scenario == "shell_timeout":
        from my_assistant.sandbox.local.local_sandbox import LocalSandbox

        output = LocalSandbox(id="benchmark").execute_command('python -c "import time; time.sleep(2)"', timeout=0.1)
        passed = "timed out" in output.casefold() and "Exit Code: 124" in output
        evidence = {"timeout_notice": "timed out" in output.casefold(), "exit_code_124": "Exit Code: 124" in output}
    return {
        "case_id": item["metadata"]["case_id"],
        "category": item["metadata"]["category"],
        "success": passed,
        "safety_violation": False,
        "assertions": [{"passed": passed, "detail": json.dumps(evidence, ensure_ascii=False, default=str)}],
        "response": "",
        "tool_names": [],
        "latency_ms": (time.perf_counter() - started) * 1000,
        "usage": {},
        "evidence": evidence,
    }


def _dispose_runtime_memory() -> None:
    from my_assistant.agents.memory import manager as memory_module

    manager = memory_module._memory_manager
    if manager is not None:
        try:
            manager.shutdown_flush(timeout=0.2)
        finally:
            manager.close()
    memory_module.reset_memory_manager()


def _task_factory(
    sandbox_root: Path, run_key: str, *, recursion_limit: int | None = None
):
    async def task(*, item: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            item_run_key = f"{run_key}:repeat-{item['metadata'].get('repeat', 1)}"
            if item["input"]["mode"] == "agent":
                case_home = sandbox_root / _safe_id(f"{item_run_key}:{item['metadata']['case_id']}") / "app-home"
                case_home.mkdir(parents=True, exist_ok=True)
                os.environ["MY_ASSISTANT_HOME"] = str(case_home)
                try:
                    return await asyncio.to_thread(
                        _agent_case,
                        item,
                        sandbox_root,
                        item_run_key,
                        recursion_limit=recursion_limit,
                    )
                finally:
                    await asyncio.to_thread(_dispose_runtime_memory)
            return await asyncio.to_thread(_harness_case, item, sandbox_root, item_run_key)
        except Exception as exc:  # noqa: BLE001 - Keep a failed case visible in the Langfuse run.
            return {
                "case_id": item["metadata"]["case_id"],
                "category": item["metadata"]["category"],
                "success": False,
                "safety_violation": False,
                "assertions": [{"passed": False, "detail": f"{type(exc).__name__}: {str(exc)[:500]}"}],
                "response": "",
                "tool_names": [],
                "tool_results": [],
                "latency_ms": (time.perf_counter() - started) * 1000,
                "usage": {},
                "error_type": type(exc).__name__,
            }

    return task


def _metric_score(metric: str, output: dict[str, Any]) -> float:
    if metric == "workspace_violation_rate":
        return float(bool(output.get("safety_violation")))
    return float(bool(output.get("success")))


def _item_evaluator(*, input: Any, output: dict[str, Any], expected_output: Any, metadata: dict[str, Any], **_kwargs: Any):
    from langfuse import Evaluation

    scores = [
        Evaluation(name="task_success", value=float(bool(output.get("success"))), comment="All case assertions must pass."),
        Evaluation(name="safety_violation", value=bool(output.get("safety_violation")), data_type="BOOLEAN"),
        Evaluation(name="latency_ms", value=float(output.get("latency_ms", 0.0))),
        Evaluation(name="tool_call_count", value=float(len(output.get("tool_names", [])))),
    ]
    usage = output.get("usage", {})
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        if isinstance(usage.get(key), int):
            scores.append(Evaluation(name=key, value=float(usage[key])))
    metric = metadata.get("resume_metric")
    if metric:
        scores.append(Evaluation(name=metric, value=_metric_score(metric, output)))
    return scores


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * percentile)))
    return float(ordered[index])


def _run_evaluator(*, item_results: list[Any], **_kwargs: Any):
    from langfuse import Evaluation

    rows = []
    for result in item_results:
        item = result.item if isinstance(result.item, dict) else {}
        metadata = item.get("metadata", {})
        output = result.output if isinstance(result.output, dict) else {}
        rows.append({"metadata": metadata, "output": output})
    evaluations = []
    for category in ("files", "dialogue", "memory", "tools"):
        values = [row["output"].get("success", False) for row in rows if row["metadata"].get("category") == category]
        if values:
            evaluations.append(Evaluation(name=f"{category}_success_rate", value=sum(values) / len(values), comment=f"{sum(values)}/{len(values)} cases passed; synthetic benchmark only."))
    agent_latencies = [
        float(row["output"].get("latency_ms", 0))
        for row in rows
        if row["metadata"].get("track") == "agent" and row["output"].get("latency_ms", 0) > 0
    ]
    if agent_latencies:
        evaluations.extend([
            Evaluation(name="agent_latency_p50_ms", value=_percentile(agent_latencies, 0.50)),
            Evaluation(name="agent_latency_p95_ms", value=_percentile(agent_latencies, 0.95)),
        ])
    all_cases = [row["output"] for row in rows]
    evaluations.append(Evaluation(name="safety_violation_rate", value=sum(bool(output.get("safety_violation")) for output in all_cases) / max(1, len(all_cases))))
    metrics = {row["metadata"].get("resume_metric") for row in rows if row["metadata"].get("resume_metric")}
    for metric in sorted(metrics):
        values = [
            _metric_score(metric, row["output"])
            for row in rows
            if row["metadata"].get("resume_metric") == metric
        ]
        if metric == "workspace_violation_rate":
            comment = f"{sum(values)}/{len(values)} applicable synthetic cases violated the workspace boundary."
        else:
            comment = f"{sum(values)}/{len(values)} applicable synthetic cases passed."
        evaluations.append(Evaluation(name=metric, value=sum(values) / len(values), comment=comment))
    token_totals = [int(row["output"].get("usage", {}).get("total_tokens", 0)) for row in rows if row["metadata"].get("track") == "agent" and row["output"].get("usage", {}).get("total_tokens")]
    if token_totals:
        evaluations.append(Evaluation(name="agent_mean_tokens_per_case", value=statistics.mean(token_totals)))
    return evaluations


def _build_report(
    result: Any,
    *,
    run_key: str,
    case_count: int,
    repeats: int,
    recursion_limit: int,
) -> dict[str, Any]:
    rows = []
    for item_result in result.item_results:
        item = item_result.item if isinstance(item_result.item, dict) else {}
        output = item_result.output if isinstance(item_result.output, dict) else {}
        rows.append({
            "case_id": output.get("case_id") or item.get("metadata", {}).get("case_id"),
            "category": output.get("category") or item.get("metadata", {}).get("category"),
            "success": bool(output.get("success")),
            "safety_violation": bool(output.get("safety_violation")),
            "latency_ms": output.get("latency_ms"),
            "tokens": output.get("usage", {}),
            "tool_names": output.get("tool_names", []),
            "assertions": output.get("assertions", []),
            "error_type": output.get("error_type"),
            "trace_id": item_result.trace_id,
            "track": item.get("metadata", {}).get("track", "harness"),
            "resume_metric": item.get("metadata", {}).get("resume_metric"),
        })
    summaries: dict[str, Any] = {}
    summaries["overall_success_rate"] = {
        "value": sum(row["success"] for row in rows) / max(1, len(rows)),
        "passed": sum(row["success"] for row in rows),
        "total": len(rows),
    }
    for category in ("files", "dialogue", "memory", "tools"):
        selected = [row for row in rows if row["category"] == category]
        summaries[f"{category}_success_rate"] = {
            "value": sum(row["success"] for row in selected) / max(1, len(selected)),
            "passed": sum(row["success"] for row in selected),
            "total": len(selected),
        }
    eligible = [row for row in rows if row["case_id"] == "files.workspace_boundary"]
    summaries["workspace_violation_rate"] = {
        "value": sum(row["safety_violation"] for row in eligible) / max(1, len(eligible)),
        "violations": sum(row["safety_violation"] for row in eligible),
        "attempts": len(eligible),
    }
    durations = [float(row["latency_ms"]) for row in rows if row["latency_ms"] and row["track"] == "agent"]
    summaries["agent_latency_ms"] = {"p50": _percentile(durations, .5), "p95": _percentile(durations, .95), "n": len(durations)}
    behavior_metrics: dict[str, dict[str, Any]] = {}
    for row in rows:
        metric = row.get("resume_metric")
        if metric and metric != "workspace_violation_rate":
            aggregate = behavior_metrics.setdefault(metric, {"passed": 0, "total": 0, "value": 0.0})
            aggregate["passed"] += int(row["success"])
            aggregate["total"] += 1
    for aggregate in behavior_metrics.values():
        aggregate["value"] = aggregate["passed"] / aggregate["total"]
    summaries["behavior_metrics"] = behavior_metrics
    token_cases = [row for row in rows if row["track"] == "agent" and row["tokens"].get("total_tokens")]
    summaries["agent_tokens_per_case"] = {
        "mean": statistics.mean(int(row["tokens"]["total_tokens"]) for row in token_cases) if token_cases else None,
        "n": len(token_cases),
    }
    return {
        "benchmark": "my_assistant.personal_assistant.v1",
        "run_key": run_key,
        "created_at": datetime.now(UTC).isoformat(),
        "case_count_per_repeat": case_count,
        "repeats": repeats,
        "total_items": len(rows),
        "model": os.environ.get("MY_ASSISTANT_BENCHMARK_MODEL", "deepseek-flash"),
        "effective_recursion_limit": recursion_limit,
        "langfuse_experiment": {"name": result.run_name, "id": result.experiment_id, "url": result.experiment_url},
        "metrics": summaries,
        "scores": {evaluation.name: evaluation.value for evaluation in result.run_evaluations},
        "items": rows,
        "interpretation": "Synthetic engineering benchmark, not production traffic or a statistically powered user study. One-case metrics must be reported with their denominator.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MyAssistant personal assistant benchmark using Langfuse experiments.")
    parser.add_argument("--dry-run", action="store_true", help="Validate cases and print their distribution without model or Langfuse calls.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat the complete suite; defaults to one pass.")
    parser.add_argument("--limit", type=int, help="Run only the first N cases (for a smoke test).")
    parser.add_argument("--recursion-limit", type=int, help="Override the agent recursion limit for an explicit comparison run.")
    parser.add_argument("--run-name", help="Override the Langfuse experiment run name.")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.recursion_limit is not None and args.recursion_limit < 1:
        parser.error("--recursion-limit must be at least 1")

    import yaml

    app_config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    configured_limit = int(app_config.get("recursion_limit", 100))
    recursion_ceiling = int(app_config.get("max_recursion_limit", 1000))
    effective_recursion_limit = min(
        args.recursion_limit if args.recursion_limit is not None else configured_limit,
        recursion_ceiling,
    )

    cases = _load_cases()
    if args.limit is not None:
        cases = cases[: max(0, args.limit)]
    distribution: dict[str, int] = {}
    for case in cases:
        category = case["metadata"]["category"]
        distribution[category] = distribution.get(category, 0) + 1
    print(json.dumps({"cases_per_repeat": len(cases), "repeat": args.repeat, "distribution": distribution, "effective_recursion_limit": effective_recursion_limit}, ensure_ascii=False))
    if args.dry_run:
        return 0

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    required = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "DEEPSEEK_API_KEY")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"Missing required environment variable(s): {', '.join(missing)}")
    os.environ["LANGFUSE_TRACING"] = "true"
    os.environ.setdefault("MY_ASSISTANT_PROJECT_ROOT", str(ROOT))
    run_key = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid_suffix()
    data = []
    for repeat_index in range(args.repeat):
        for source in cases:
            item = json.loads(json.dumps(source))
            item["metadata"]["repeat"] = repeat_index + 1
            item["metadata"]["track"] = item["metadata"].get("track", "harness")
            data.append(item)

    with tempfile.TemporaryDirectory(prefix="my-assistant-benchmark-", ignore_cleanup_errors=True) as temporary:
        sandbox_root = Path(temporary) / "runtime"
        sandbox_root.mkdir(parents=True, exist_ok=True)
        os.environ["MY_ASSISTANT_HOME"] = str(sandbox_root / "app-home")
        from langfuse import get_client

        langfuse = get_client()
        report_name = args.run_name or f"personal-assistant-benchmark-{run_key}"
        result = langfuse.run_experiment(
            name="Personal Assistant Business Scenarios",
            run_name=report_name,
            description="Synthetic file, dialogue continuity, memory, and tool reliability benchmark for MyAssistant Harness.",
            data=data,
            task=_task_factory(
                sandbox_root, run_key, recursion_limit=effective_recursion_limit
            ),
            evaluators=[_item_evaluator],
            run_evaluators=[_run_evaluator],
            max_concurrency=1,
            metadata={"benchmark": "my_assistant.personal_assistant.v1", "model": os.environ.get("MY_ASSISTANT_BENCHMARK_MODEL", "deepseek-flash"), "source": "synthetic-local-cases", "recursion_limit": str(effective_recursion_limit)},
        )
        langfuse.flush()
        report = _build_report(
            result,
            run_key=run_key,
            case_count=len(cases),
            repeats=args.repeat,
            recursion_limit=effective_recursion_limit,
        )

    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / f"{run_key}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (results_dir / "latest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"items": len(report["items"]), "metrics": report["metrics"]}, ensure_ascii=True))
    print(f"Local report: {report_path}")
    if result.experiment_url:
        print(f"Langfuse experiment: {result.experiment_url}")
    return 0


def uuid_suffix() -> str:
    return hashlib.sha256(os.urandom(16)).hexdigest()[:8]


if __name__ == "__main__":
    raise SystemExit(main())
