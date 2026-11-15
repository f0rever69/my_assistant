from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from my_assistant.agents.memory.backends.mymem.my_mem import MyMem
from my_assistant.agents.memory.backends.mymem.mymem.core.storage import (
    create_empty_memory,
)
from my_assistant.agents.memory.policy import fact_expired, is_sensitive_memory_text
from my_assistant.tui.app import MyAssistantTUI


def _manager(tmp_path, **overrides) -> MyMem:
    config = {
        "storage_path": str(tmp_path),
        "retrieval_adapter": "",
        "fact_ttl_days": 365,
    }
    config.update(overrides)
    return MyMem(backend_config=config)


def test_sensitive_facts_are_rejected_by_policy_and_crud(tmp_path) -> None:
    manager = _manager(tmp_path)

    assert is_sensitive_memory_text("My API key is sk-test-value")
    assert not is_sensitive_memory_text("I mainly use Python for automation")
    with pytest.raises(ValueError, match="Sensitive facts"):
        manager.create_fact("My password is hunter2", user_id="u1")
    with pytest.raises(ValueError, match="Sensitive data"):
        manager.import_memory(
            {
                "user": {
                    "personalContext": {
                        "summary": "The user's API key is sk-abcdefghijklmnopqrstuvwxyz"
                    }
                }
            },
            user_id="u1",
            agent_name="test-agent",
        )


def test_user_correction_replaces_old_truth_and_keeps_audit_metadata(tmp_path) -> None:
    manager = _manager(tmp_path)
    _, fact_id = manager.create_fact(
        "The user mainly uses Python",
        confidence=0.7,
        user_id="u1",
    )
    assert fact_id

    manager.update_fact(
        fact_id,
        content="The user now mainly uses Go",
        category="correction",
        confidence=1.0,
        source_thread_id="thread-42",
        user_id="u1",
    )

    fact = manager.get_memory(user_id="u1")["facts"][0]
    assert fact["content"] == "The user now mainly uses Go"
    assert fact["previousContent"] == "The user mainly uses Python"
    assert fact["previousSource"] == {"type": "manual", "threadId": None}
    assert fact["evidenceType"] == "user_explicit"
    assert fact["sourceThreadId"] == "thread-42"
    assert fact["updatedAt"] >= fact["createdAt"]
    assert manager.search("Python", user_id="u1") == []


def test_hard_ttl_marks_old_fact_inactive(tmp_path) -> None:
    manager = _manager(tmp_path, fact_ttl_days=1)
    old = (datetime.now(UTC) - timedelta(days=2)).isoformat().replace("+00:00", "Z")
    manager.import_memory(
        {
            "facts": [
                {
                    "id": "fact_old",
                    "content": "The user once used Pascal",
                    "category": "context",
                    "confidence": 0.8,
                    "createdAt": old,
                    "ttlDays": 1,
                    "source": "thread-old",
                }
            ]
        },
        user_id="u1",
        agent_name="test-agent",
    )

    assert manager.get_memory(user_id="u1", agent_name="test-agent")["facts"] == []
    assert fact_expired({"createdAt": old, "ttlDays": 1})
    # Expiry removes a fact from normal reads but leaves it addressable for an
    # explicit forget operation and audit cleanup.
    manager.delete_fact("fact_old", user_id="u1", agent_name="test-agent")
    assert manager._updater.get_memory_data(
        "test-agent", user_id="u1"
    )["facts"] == []


def test_automatic_conflict_requires_and_applies_valid_replacement(tmp_path) -> None:
    manager = _manager(tmp_path)
    current = create_empty_memory()
    current["facts"] = [
        {
            "id": "fact_old",
            "content": "The user mainly uses Python",
            "category": "preference",
            "confidence": 0.9,
            "createdAt": "2026-01-01T00:00:00Z",
            "source": "thread-old",
        }
    ]
    update = {
        "newFacts": [
            {
                "content": "The user now mainly uses Go",
                "category": "correction",
                "confidence": 0.99,
                "evidenceType": "user_explicit",
                "scope": "user",
                "durability": "durable",
                "authority": "descriptive",
            }
        ],
        "factsToRemove": [
            {
                "id": "fact_old",
                "scope": "user",
                "reason": "explicit user correction",
                "replacementFactIndex": 0,
            }
        ],
    }

    result = manager._updater._apply_updates(
        current,
        update,
        thread_id="thread-new",
        agent_name="__default__",
        user_id="u1",
    )

    assert [fact["content"] for fact in result["facts"]] == ["The user now mainly uses Go"]
    assert result["facts"][0]["evidenceType"] == "user_explicit"
    assert result["facts"][0]["source"] == {"type": "conversation", "threadId": "thread-new"}


def test_temporary_and_sensitive_extraction_never_become_long_term_memory(tmp_path) -> None:
    manager = _manager(tmp_path)
    current = create_empty_memory()
    update = {
        "user": {
            "workContext": {
                "summary": "The user's API key is sk-abcdefghijklmnopqrstuvwxyz",
                "shouldUpdate": True,
                "scope": "user",
                "authority": "descriptive",
            }
        },
        "newFacts": [
            {
                "content": "Use Python only for this answer",
                "category": "preference",
                "confidence": 0.99,
                "evidenceType": "user_explicit",
                "scope": "thread",
                "durability": "temporary",
                "authority": "descriptive",
            },
            {
                "content": "The user's API key is sk-abcdefghijklmnopqrstuvwxyz",
                "category": "context",
                "confidence": 0.99,
                "evidenceType": "user_explicit",
                "scope": "user",
                "durability": "durable",
                "authority": "descriptive",
            },
        ],
    }

    result = manager._updater._apply_updates(
        current,
        update,
        thread_id="thread-new",
        agent_name="__default__",
        user_id="u1",
    )

    assert result["facts"] == []
    assert result["user"]["workContext"]["summary"] == ""


class _TuiClient:
    def __init__(self) -> None:
        self.facts = [
            {
                "id": "fact_1",
                "content": "Uses Python",
                "sourceThreadId": "thread-original",
                "updatedAt": "2026-01-01T00:00:00Z",
                "evidenceType": "model_inferred",
            }
        ]
        self.updated: tuple | None = None
        self.deleted: str | None = None

    def get_memory(self) -> dict:
        return {"facts": self.facts}

    def update_memory_fact(self, fact_id: str, **kwargs) -> None:
        self.updated = (fact_id, kwargs)

    def delete_memory_fact(self, fact_id: str) -> None:
        self.deleted = fact_id


def test_tui_memory_commands_list_search_correct_and_forget() -> None:
    client = _TuiClient()
    events = []
    fake_app = SimpleNamespace(
        session=SimpleNamespace(client=client),
        _dispatch=events.append,
        _conv_thread_id="thread-current",
    )

    MyAssistantTUI._show_memory(fake_app, "list")
    assert "thread-original" in events[-1].text
    assert "model_inferred" in events[-1].text

    MyAssistantTUI._show_memory(fake_app, "search Python")
    assert "Uses Python" in events[-1].text

    MyAssistantTUI._show_memory(fake_app, 'correct fact_1 "Now uses Go"')
    assert client.updated == (
        "fact_1",
        {
            "content": "Now uses Go",
            "category": "correction",
            "confidence": 1.0,
            "source_thread_id": "thread-current",
        },
    )

    MyAssistantTUI._show_memory(fake_app, "forget fact_1")
    assert client.deleted == "fact_1"
