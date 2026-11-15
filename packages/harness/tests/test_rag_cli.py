import json
import socket
from types import SimpleNamespace

from my_assistant.rag import local
from my_assistant.tui.app import MyAssistantTUI
from my_assistant.tui.cli import main
from my_assistant.tui.command_registry import resolve
from my_assistant.tui.view_state import RunEnded, SystemMessage


def test_cli_mock_runs_real_retrieval_and_audit_offline(tmp_path, monkeypatch, capsys):
    def unexpected_network(*args, **kwargs):
        raise AssertionError("unexpected network attempt")

    monkeypatch.setattr(socket.socket, "connect", unexpected_network)
    monkeypatch.setattr(socket, "getaddrinfo", unexpected_network)
    monkeypatch.setenv("MY_ASSISTANT_HOME", str(tmp_path / "runtime"))
    root = tmp_path / "workspace"
    root.mkdir()
    source = root / "制度.txt"
    quote = "报销期限为30天。"
    source.write_text(quote, encoding="utf-8")
    base = ["kb", "--workspace", str(root)]
    assert main([*base, "index", "--source", "."]) == 0
    assert json.loads(capsys.readouterr().out)["indexed"] == 1
    assert main([*base, "search", "报销期限"]) == 0
    [hit] = json.loads(capsys.readouterr().out)["hits"]
    fixture = tmp_path / "mock.json"
    fixture.write_text(
        json.dumps(
            {
                "extract": {
                    "topics": [
                        {
                            "question": "报销期限",
                            "claims": [
                                {
                                    "text": quote,
                                    "quotes": [
                                        {"chunk_id": hit["chunk_id"], "quote": quote}
                                    ],
                                }
                            ],
                        }
                    ]
                },
                "verify": {
                    "judgments": [
                        {
                            "claim_id": "1",
                            "verdict": "supported",
                            "evidence_ids": ["1.1"],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    assert main([*base, "ask", "报销期限", "--mock", str(fixture), "--json"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["findings"][0]["citations"][0]["quote"] == quote
    assert main([*base, "audit"]) == 0
    [report] = json.loads(capsys.readouterr().out)
    assert report["trace_id"] == answer["trace_id"]
    source.write_text("报销期限为15天。", encoding="utf-8")
    assert main([*base, "ask", "报销期限", "--mock", str(fixture)]) == 0
    assert "证据不足" in capsys.readouterr().out


def test_tui_worker_uses_knowledge_service_and_releases_run(monkeypatch):
    assert resolve("/askkb 查资料", skills={"askkb"}).kind == "builtin"
    index = object()
    calls = []
    service = SimpleNamespace(
        ask=lambda question, mode: SimpleNamespace(
            render=lambda: f"{question} ({mode})",
            error=None,
            trace_id="trace",
        )
    )
    monkeypatch.setattr(local, "open_index", lambda workspace_id: index)
    monkeypatch.setattr(
        local,
        "open_service",
        lambda received: (service, SimpleNamespace(output_mode="strict")),
    )
    receiver = SimpleNamespace(
        _run_token="token",
        _on_run_action=lambda token, action: calls.append((token, action)),
        call_from_thread=lambda fn, *args: fn(*args),
    )
    MyAssistantTUI._knowledge_worker(receiver, "查资料", "workspace", "token")
    assert calls[0] == ("token", SystemMessage("查资料 (strict)\ntrace_id: trace"))
    assert calls[1] == ("token", RunEnded())
