from __future__ import annotations

import logging
import os
import re
from typing import Any

from my_assistant.config import (
    get_enabled_tracing_providers,
    get_tracing_config,
    is_monocle_tracing_enabled,
    validate_enabled_tracing_providers,
)
from my_assistant.tracing.monocle import is_monocle_setup_completed

logger = logging.getLogger(__name__)


def _create_langsmith_tracer(config) -> Any:
    from langchain_core.tracers.langchain import LangChainTracer

    return LangChainTracer(project_name=config.project)


def _resolve_langfuse_environment() -> str | None:
    environment = (
        os.environ.get("LANGFUSE_TRACING_ENVIRONMENT")
        or os.environ.get("MY_ASSISTANT_ENV")
        or os.environ.get("ENVIRONMENT")
    )
    if not environment:
        return None

    normalized = re.sub(r"[^a-z0-9_-]+", "-", environment.strip().lower()).strip("-_")
    if not normalized or normalized.startswith("langfuse") or len(normalized) > 40:
        return None
    return normalized


def _create_langfuse_handler(config) -> Any:
    from langfuse import Langfuse

    from my_assistant.tracing.langfuse_handler import (
        MyAssistantLangfuseCallbackHandler,
        mask_langfuse_data,
    )

    environment = _resolve_langfuse_environment()
    # Langfuse keeps one configured client per public key. Install the mask
    # before creating the callback so every observation is filtered at capture.
    Langfuse(
        secret_key=config.secret_key,
        public_key=config.public_key,
        base_url=config.host,
        environment=environment,
        mask=mask_langfuse_data,
    )
    return MyAssistantLangfuseCallbackHandler(public_key=config.public_key)


def build_tracing_callbacks() -> list[Any]:
    """Build callbacks for all explicitly enabled tracing providers."""
    validate_enabled_tracing_providers()
    # Monocle is not a callback provider; this per-run path is just where an
    # embedded process that skipped Gateway-lifespan setup can be told about it.
    if is_monocle_tracing_enabled() and not is_monocle_setup_completed():
        logger.debug(
            "MONOCLE_TRACING is set but Monocle is not initialized in this process — only the Gateway lifespan runs setup automatically; embedded/TUI callers must call my_assistant.tracing.setup_monocle_tracing_if_enabled() themselves."
        )
    enabled_providers = get_enabled_tracing_providers()
    if not enabled_providers:
        return []

    tracing_config = get_tracing_config()
    callbacks: list[Any] = []

    for provider in enabled_providers:
        if provider == "langsmith":
            try:
                callbacks.append(_create_langsmith_tracer(tracing_config.langsmith))
            except (
                Exception
            ) as exc:  # pragma: no cover - exercised via tests with monkeypatch
                raise RuntimeError(
                    f"LangSmith tracing initialization failed: {exc}"
                ) from exc
        elif provider == "langfuse":
            try:
                callbacks.append(_create_langfuse_handler(tracing_config.langfuse))
            except (
                Exception
            ) as exc:  # pragma: no cover - exercised via tests with monkeypatch
                raise RuntimeError(
                    f"Langfuse tracing initialization failed: {exc}"
                ) from exc

    return callbacks
