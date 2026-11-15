"""Public contracts for MyAssistant extensions.

This package MUST NOT import `my_assistant`. Every host contract an extension
needs lives here, while framework imports remain direct extension dependencies;
extensions can therefore be released independently of the host.
"""

from __future__ import annotations

from my_assistant_extension_api.assembly import (
    AgentAssemblyDescriptor,
    AgentAssemblyObserver,
    MiddlewareDescriptor,
    ToolDescriptor,
)
from my_assistant_extension_api.compaction import (
    CompactionEvent,
    ContextCompactionObserver,
)
from my_assistant_extension_api.contracts import (
    ExtensionInstall,
    ExtensionRegistry,
    HostPolicySnapshot,
    MiddlewareContributor,
    SystemModelCallObserver,
    SystemModelRequest,
    SystemModelResult,
    SystemOperationKind,
    TaskInfo,
    TaskLifecycleContributor,
    TaskOutcome,
    extension,
)
from my_assistant_extension_api.placement import (
    AgentBuildContext,
    AgentScope,
    MiddlewarePlacement,
    Placement,
)
from my_assistant_extension_api.provenance import (
    MESSAGE_CONTENT_KIND_KEY,
    MESSAGE_PRODUCER_ENTITY_ID_KEY,
    MESSAGE_PRODUCER_KIND_KEY,
    PROVENANCE_KEYS,
    ContentKind,
    MessageProvenance,
    provenance_kwargs,
    read_provenance,
)
from my_assistant_extension_api.release import (
    ReleasePolicyProvider,
    canonical_hash,
    canonical_json,
    collect_release_policies,
)
from my_assistant_extension_api.run_evidence import (
    InvalidRunEvidenceCursor,
    RunEventPage,
    RunEventView,
    RunEvidenceReader,
    RunPage,
    RunStatusView,
)
from my_assistant_extension_api.runtime_bridge import (
    EXTENSION_TASK_STORE_KEY,
    task_store_from_runtime,
)
from my_assistant_extension_api.state import ExtensionData

#: Contract version. Before 1.0, minors may break and patches are additive.
#: From 1.0 on, bump the major for breaking changes.
API_VERSION = "0.3.0"

__all__ = [
    "API_VERSION",
    "EXTENSION_TASK_STORE_KEY",
    "MESSAGE_CONTENT_KIND_KEY",
    "MESSAGE_PRODUCER_ENTITY_ID_KEY",
    "MESSAGE_PRODUCER_KIND_KEY",
    "PROVENANCE_KEYS",
    "AgentAssemblyDescriptor",
    "AgentAssemblyObserver",
    "AgentBuildContext",
    "AgentScope",
    "CompactionEvent",
    "ContentKind",
    "ContextCompactionObserver",
    "ExtensionData",
    "ExtensionInstall",
    "ExtensionRegistry",
    "HostPolicySnapshot",
    "InvalidRunEvidenceCursor",
    "MessageProvenance",
    "MiddlewareContributor",
    "MiddlewareDescriptor",
    "MiddlewarePlacement",
    "Placement",
    "ReleasePolicyProvider",
    "RunEvidenceReader",
    "RunEventPage",
    "RunEventView",
    "RunPage",
    "RunStatusView",
    "SystemModelCallObserver",
    "SystemModelRequest",
    "SystemModelResult",
    "SystemOperationKind",
    "TaskInfo",
    "TaskLifecycleContributor",
    "TaskOutcome",
    "ToolDescriptor",
    "canonical_hash",
    "canonical_json",
    "collect_release_policies",
    "extension",
    "provenance_kwargs",
    "read_provenance",
    "task_store_from_runtime",
]
