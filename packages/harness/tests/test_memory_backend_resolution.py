from __future__ import annotations

import pytest
from my_assistant.agents.memory import manager as memory_manager
from my_assistant.agents.memory.backends.mymem import MANAGER_CLASS
from my_assistant.agents.memory.backends.mymem.my_mem import MyMem
from my_assistant.agents.memory.backends.mymem.mymem.config import MyMemConfig
from my_assistant.agents.memory.backends.mymem.mymem.core import paths as memory_paths
from my_assistant.config.memory_config import MemoryConfig


@pytest.fixture
def backend_factory(monkeypatch):
    """Keep factory resolution tests isolated from application singletons."""
    monkeypatch.setattr(memory_manager, "_memory_manager", None)
    monkeypatch.setattr(memory_manager, "_backends_cache", None)
    monkeypatch.setattr(memory_manager, "_collect_host_hooks", dict)
    yield memory_manager
    if memory_manager._memory_manager is not None:
        memory_manager._memory_manager.close()
    memory_manager._memory_manager = None
    memory_manager._backends_cache = None


@pytest.mark.parametrize(
    "selector",
    [
        MemoryConfig().manager_class,
        "my_assistant.agents.memory.backends.mymem.my_mem:MyMem",
        "my_assistant.agents.memory.backends.mymem.my_mem.MyMem",
    ],
)
def test_backend_factory_reopens_existing_memory(
    backend_factory, monkeypatch, tmp_path, selector
):
    monkeypatch.setenv("MY_ASSISTANT_HOME", str(tmp_path))
    config = MemoryConfig(
        manager_class=selector,
        backend_config={"retrieval_adapter": "", "token_counting": "char"},
    )
    monkeypatch.setattr(backend_factory, "get_memory_config", lambda: config)

    first = backend_factory.get_memory_manager()
    assert isinstance(first, MyMem)
    assert first.backend_config["storage_path"] == str(tmp_path)
    _, fact_id = first.create_fact("The user mainly uses Python", user_id="test-user")
    assert fact_id
    first.close()
    backend_factory.reset_memory_manager()

    reopened = backend_factory.get_memory_manager()
    facts = reopened.get_memory(user_id="test-user")["facts"]
    assert any(
        fact["id"] == fact_id and fact["content"] == "The user mainly uses Python"
        for fact in facts
    )
    assert (tmp_path / "users" / "test-user" / "memory.json").is_file()


def test_default_discovers_mymem_backend(backend_factory):
    assert MemoryConfig().manager_class == "mymem"
    registry = backend_factory._scan_backends()
    assert "mymem" in registry
    assert registry["mymem"] is MANAGER_CLASS is MyMem
    assert (
        backend_factory._resolve_manager_class("mymem", allow_discovery=False) is MyMem
    )


def test_unknown_backend_still_fails_closed(backend_factory):
    with pytest.raises(ValueError, match="not a registered backend name"):
        backend_factory._resolve_manager_class("myemm")


def test_standalone_storage_uses_mymem_environment_and_home(monkeypatch, tmp_path):
    monkeypatch.delenv("MYMEM_DATA_DIR", raising=False)
    monkeypatch.setattr(memory_paths.Path, "home", classmethod(lambda cls: tmp_path))
    config = MyMemConfig(token_counting="char")
    assert memory_paths.memory_file_path(config) == tmp_path / ".mymem" / "memory.json"

    data_root = tmp_path / "custom-memory"
    monkeypatch.setenv("MYMEM_DATA_DIR", str(data_root))
    assert (
        memory_paths.memory_file_path(config, user_id="u1")
        == data_root / "users" / "u1" / "memory.json"
    )

    explicit_root = tmp_path / "explicit-memory"
    config.storage_path = str(explicit_root)
    assert (
        memory_paths.memory_file_path(config, user_id="u1")
        == explicit_root / "users" / "u1" / "memory.json"
    )
