import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from my_assistant.community.aio_sandbox import AioSandboxProvider
from my_assistant.community.aio_sandbox.local_backend import LocalContainerBackend


@pytest.fixture
def orphaned_backend():
    backend = LocalContainerBackend.__new__(LocalContainerBackend)
    backend._runtime = "docker"
    backend._network_mode = "isolated"
    backend._container_prefix = "test-sandbox"
    backend._is_container_running = Mock(return_value=False)
    backend._batch_inspect = Mock(return_value={})
    backend.destroy = Mock()
    return backend


@pytest.mark.parametrize("role", ["network", "egress-network", "network-proxy"])
def test_discover_reports_remnants_for_fenced_replacement(orphaned_backend, role):
    sandbox_id = "orphan-test"
    labels = {"my_assistant.sandbox_id": sandbox_id, "my_assistant.role": role}
    resource = SimpleNamespace(labels=labels)
    proxy, network = orphaned_backend._resource_names(sandbox_id)
    egress = orphaned_backend._egress_network_name(sandbox_id)
    orphaned_backend._inspect_network = Mock(
        side_effect=lambda name: (
            resource
            if (name, role) in {(network, "network"), (egress, "egress-network")}
            else None
        )
    )
    if role == "network-proxy":
        orphaned_backend._batch_inspect.return_value = {proxy: resource}

    info = orphaned_backend.discover(sandbox_id)

    assert info is not None and info.requires_replacement
    assert info.sandbox_id == sandbox_id and not info.sandbox_url
    orphaned_backend.destroy.assert_not_called()


def test_discovery_does_not_replace_foreign_resources(orphaned_backend):
    foreign = SimpleNamespace(labels={"my_assistant.sandbox_id": "other"})
    orphaned_backend._inspect_network = Mock(return_value=foreign)
    assert orphaned_backend.discover("orphan-test") is None
    orphaned_backend.destroy.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("can_replace", [False, True])
def test_provider_fences_replacement_before_create(
    tmp_path, monkeypatch, asynchronous, can_replace
):
    from my_assistant.community.aio_sandbox.aio_sandbox_provider import (
        SandboxPolicyReplacementDeferredError,
    )
    from my_assistant.community.aio_sandbox.sandbox_info import SandboxInfo
    from my_assistant.config import paths

    monkeypatch.setenv("MY_ASSISTANT_HOME", str(tmp_path))
    monkeypatch.setattr(paths, "_paths", None)
    info = SandboxInfo("orphan-test", "", requires_replacement=True)
    provider = AioSandboxProvider.__new__(AioSandboxProvider)
    calls = []
    provider._backend = SimpleNamespace(discover=lambda _: info)
    provider._recheck_cached_sandbox = lambda *args, **kwargs: None

    def replace_resource(*args):
        calls.append("fence")
        return can_replace

    def create_resource(*args, **kwargs):
        calls.append("create")
        assert calls == ["fence", "create"]
        return info.sandbox_id

    async def create_async(*args, **kwargs):
        return create_resource()

    provider._replace_incompatible_sandbox = replace_resource
    provider._create_sandbox = create_resource
    provider._create_sandbox_async = create_async

    def acquire():
        if asynchronous:
            return asyncio.run(
                provider._discover_or_create_with_lock_async(
                    "thread-test", info.sandbox_id
                )
            )
        return provider._discover_or_create_with_lock("thread-test", info.sandbox_id)

    if can_replace:
        assert acquire() == info.sandbox_id
    else:
        with pytest.raises(SandboxPolicyReplacementDeferredError):
            acquire()
        assert calls == ["fence"]
