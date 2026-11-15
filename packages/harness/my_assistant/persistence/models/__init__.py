"""ORM model registration entry point.

Importing this module ensures all ORM models are registered with
``Base.metadata`` so Alembic autogenerate detects every table.

The actual ORM classes have moved to entity-specific subpackages:
- ``my_assistant.persistence.thread_meta``
- ``my_assistant.persistence.run``
- ``my_assistant.persistence.feedback``
- ``my_assistant.persistence.user``

``RunEventRow`` remains in ``my_assistant.persistence.models.run_event`` because
its storage implementation lives in ``my_assistant.runtime.events.store.db`` and
there is no matching entity directory.
"""

from my_assistant.persistence.agents.model import AgentRow
from my_assistant.persistence.channel_connections.model import (
    ChannelConnectionRow,
    ChannelConversationRow,
    ChannelCredentialRow,
    ChannelOAuthStateRow,
)
from my_assistant.persistence.feedback.model import FeedbackRow
from my_assistant.persistence.managed_subagents.model import ManagedSubagentRow
from my_assistant.persistence.mcp_tasks.model import McpTaskRow
from my_assistant.persistence.models.run_event import RunEventRow
from my_assistant.persistence.personal_access_tokens.model import PersonalAccessTokenRow
from my_assistant.persistence.projects.model import ProjectDocumentRow, ProjectRow
from my_assistant.persistence.run.model import RunRow
from my_assistant.persistence.scheduled_task_runs.model import ScheduledTaskRunRow
from my_assistant.persistence.scheduled_tasks.model import ScheduledTaskRow
from my_assistant.persistence.subagent_batches.model import SubagentBatchItemRow, SubagentBatchRow
from my_assistant.persistence.thread_meta.model import ThreadMetaRow
from my_assistant.persistence.user.model import UserRow
from my_assistant.persistence.webhook_delivery.model import WebhookDeliveryRow

__all__ = [
    "AgentRow",
    "ChannelConnectionRow",
    "ChannelConversationRow",
    "ChannelCredentialRow",
    "ChannelOAuthStateRow",
    "FeedbackRow",
    "McpTaskRow",
    "ManagedSubagentRow",
    "PersonalAccessTokenRow",
    "ProjectDocumentRow",
    "ProjectRow",
    "RunEventRow",
    "RunRow",
    "ScheduledTaskRow",
    "ScheduledTaskRunRow",
    "SubagentBatchRow",
    "SubagentBatchItemRow",
    "ThreadMetaRow",
    "UserRow",
    "WebhookDeliveryRow",
]
