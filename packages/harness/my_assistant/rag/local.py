"""Composition shared by the local CLI and TUI."""

from __future__ import annotations

import hashlib
from pathlib import Path

from my_assistant.config.paths import get_paths
from my_assistant.config.rag_config import RagConfig
from my_assistant.runtime.user_context import get_effective_user_id
from my_assistant.workspaces import get_workspace_registry

from .audit import AuditLog
from .index import KnowledgeIndex
from .model import ChatEvidenceModel, MockEvidenceModel
from .service import KnowledgeService


def open_index(
    *, path: str | None = None, workspace_id: str | None = None
) -> KnowledgeIndex:
    user_id = get_effective_user_id()
    registry = get_workspace_registry(user_id)
    if path is not None:
        workspace = registry.register(path)
    elif workspace_id is not None:
        workspace = registry.resolve(workspace_id)
    else:
        raise ValueError("Select a registered Workspace first")
    namespace = hashlib.sha256(
        f"{workspace.id}:{workspace.root_fingerprint}".encode()
    ).hexdigest()[:32]
    db_path = get_paths().user_dir(user_id) / "knowledge" / namespace / "index.sqlite3"
    return KnowledgeIndex(
        db_path, workspace, resolve_workspace=lambda: registry.resolve(workspace.id)
    )


def audit_log(index: KnowledgeIndex, config: RagConfig) -> AuditLog:
    return AuditLog(index.db_path.with_name("audit.sqlite3"), config.audit_max_runs)


def open_service(
    index: KnowledgeIndex,
    *,
    mock_path: Path | None = None,
    allow_model_data: bool = False,
) -> tuple[KnowledgeService, RagConfig]:
    if mock_path is not None:
        config = RagConfig()
        model = MockEvidenceModel.from_file(mock_path)
        return KnowledgeService(index, model, audit=audit_log(index, config)), config

    from my_assistant.config.app_config import get_app_config
    from my_assistant.models.factory import create_chat_model

    app = get_app_config()
    config = app.rag
    if not (config.allow_model_data or allow_model_data):
        raise ValueError(
            "Model input includes selected document excerpts. Enable rag.allow_model_data or use --allow-model-data."
        )
    extraction = create_chat_model(config.model, app_config=app, attach_tracing=False)
    verifier = (
        create_chat_model(config.verifier_model, app_config=app, attach_tracing=False)
        if config.verifier_model
        else extraction
    )
    return KnowledgeService(
        index, ChatEvidenceModel(extraction, verifier), audit=audit_log(index, config)
    ), config
