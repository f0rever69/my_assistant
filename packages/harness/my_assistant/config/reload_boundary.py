"""Single source of truth for the config hot-reload boundary.

``get_app_config()`` reloads ``AppConfig`` when the config file changes, so
per-run fields take effect on the next invocation without restarting the process.
The fields listed in this module are the **infrastructure** subset captured
once at startup — engines, singletons, and logging handlers — and therefore
require a process restart to change at runtime.

The registry covers two kinds of entries:

- ``AppConfig`` fields and explicitly registered nested fields (``database``, ``checkpointer``,
  ``run_events``, ``stream_bridge``, ``sandbox``, ``log_level``). For
  these, :func:`format_field_description` produces the standardised
  ``"startup-only: ..."`` prefix that the matching Pydantic
  ``Field(description=...)`` carries, so the boundary surfaces in IDE
  hover next to the field itself.
- Nested ``AppConfig`` fields whose consumers register their own lifecycle
  boundary.

Any future "needs restart" scanner — operator tooling, lint hooks, doc
generators — should drive off this registry rather than re-parsing
prose.
"""

from __future__ import annotations

from collections.abc import Iterator

#: The standardised prefix every restart-required field description starts
#: with. ``test_reload_boundary`` enforces both directions: registered
#: fields must use this prefix in the schema, and any schema field using
#: this prefix must be in the registry.
STARTUP_ONLY_PREFIX = "startup-only:"


#: Restart-required field paths mapped to the human-readable reason.
#:
#: The reason text is what surfaces in ``Field(description=...)``, so it
#: must explain *what* code captures the snapshot — not just that the
#: field is restart-required — so an operator changing the value knows
#: which subsystem to restart.
STARTUP_ONLY_FIELDS: dict[str, str] = {
    "plugins": ("load_extensions() runs once during runtime initialization and the process-wide middleware registry is not rebuilt on config.yaml edits; adding, removing or reconfiguring a plugin requires a restart."),
    "database": ("init_engine_from_config() runs once during runtime initialization; the SQLAlchemy engine holds the connection pool and is not rebuilt on config.yaml edits."),
    "checkpointer": ("make_checkpointer() binds the persistent checkpointer once at startup, including SQLite WAL / busy_timeout settings."),
    "run_events": ("make_run_event_store() picks the memory- vs SQL-backed implementation at startup and keeps it paired with the underlying event store."),
    "agent_storage": ("runtime initialization validates agent_storage.backend against database.backend once at startup, and the db backend's synchronous SQLAlchemy engine is process-cached on first use; switching backend needs a restart."),
    "stream_bridge": ("make_stream_bridge() constructs the stream-bridge singleton once during runtime initialization."),
    "sandbox": ("get_sandbox_provider() caches the provider singleton (``_default_sandbox_provider``); a different ``sandbox.use`` class path only takes effect on next process start."),
    "skills.container_path": (
        "AioSandboxProvider and E2BSandboxProvider normalize and capture the skills mount root when their provider singleton starts; "
        "sandbox identity, mounts, remote metadata, and skill synchronization must keep using that one root until the process restarts."
    ),
    "log_level": (
        "apply_logging_level() runs only during runtime initialization; it sets logger levels and may lower root handler thresholds so configured messages can propagate. A freshly reloaded AppConfig does not retrigger it."
    ),
    "logging": (
        "configure_logging() runs only during runtime initialization; it installs/removes the trace-context filter and the enhanced formatter on root handlers, "
        "and a freshly reloaded AppConfig does not retrigger it, so a runtime edit to logging.enhance.* needs a process restart. Only log output is "
        "affected: trace ids are issued unconditionally and included in run metadata, whatever this setting says."
    ),
    "subagent_runtime": ("the shared native-subagent admission controller and isolated execution loop are configured once during runtime initialization; changing process slots, queue policy, or queue bounds requires a restart."),
    "subagent_batches": ("the durable subagent batch service is configured once during runtime initialization; capacity, leases, and recovery behavior are captured by that service instance."),
}


def iter_startup_only_field_paths() -> Iterator[str]:
    """Yield every registered restart-required field path."""
    return iter(STARTUP_ONLY_FIELDS)


def is_startup_only_field(field_path: str) -> bool:
    """Return ``True`` when *field_path* is registered as restart-required.

    Most entries are top-level paths (``"database"``, ``"sandbox"`` etc.).
    A nested path is registered only when one leaf has a different reload
    boundary from the rest of its section, such as ``skills.container_path``.
    """
    return field_path in STARTUP_ONLY_FIELDS


def format_field_description(field_path: str, *, field_doc: str | None = None) -> str:
    """Build the standardised description for a registered field.

    Used inside ``AppConfig`` ``Field(description=...)`` so the hover
    text in IDEs matches the registry and the drift tests can pin one
    side against the other.

    Args:
        field_path: A registered field path (e.g. ``"log_level"`` or
            ``"skills.container_path"``).
        field_doc: Optional human-facing description for the field itself
            (allowed values, semantics, etc.). When supplied, it is
            appended after the ``startup-only:`` marker block separated by
            a blank line so IDE hover shows both the restart-required
            reason *and* the field's normal documentation. Composition
            keeps the marker as the leading token machine-readable tooling
            pivots on while restoring the prose that ``Field(description=)``
            used to carry before the registry took over.

    Raises:
        KeyError: when *field_path* is not registered. This is deliberate
            — silently returning a placeholder would let a typo bypass
            the drift coverage.
    """
    reason = STARTUP_ONLY_FIELDS[field_path]
    header = f"{STARTUP_ONLY_PREFIX} {reason}"
    if field_doc is None:
        return header
    return f"{header}\n\n{field_doc.strip()}"
