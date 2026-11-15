# Memory Backends

Each subfolder under `agents/memory/backends/` is a pluggable memory backend. Swap the active one by changing one line in `config.yaml` - no my-assistant core changes required.

- `mymem/` - the default local backend, wrapping the self-contained MyMem engine
  (shared JSON summaries and Markdown facts).
- `noop/` - an empty backend and the **template** to copy when adding a new one.
- `openviking/` - optional remote backend using the official
  `langchain-openviking` package (single-user middleware mode).
- `honcho/` - optional remote Honcho backend over HTTP: user-model memory (representations built by Honcho's server-side deriver; no local LLM calls). Workspace-per-user isolation with per-user overrides.

This guide tells you **which files to touch** when you change, swap, or add a memory system. Paths are relative to `backend/` unless noted.

---

## Table of Contents

- [Add a New Backend](#add-a-new-backend)
- [Switch the Active Backend](#switch-the-active-backend)
- [Backend Contract](#backend-contract)
- [Do Not Modify](#do-not-modify)
- [Common Pitfalls](#common-pitfalls)
- [Reference](#reference)

## Add a New Backend

Copy `noop/` to `backends/<yourname>/` and edit three files in this folder plus two outside it.

| File | What to change |
|---|---|
| `backends/<yourname>/config.py` | Declare your config fields + `from_backend_config` (parse `backend_config`; read `storage_path` from it - **do not import my-assistant path helpers**) |
| `backends/<yourname>/<yourname>_manager.py` | Rename the class; parse config in `model_post_init`; implement `from_config` + the tier-1 abstracts (`add`/`get_context`); override tier-2/3 methods as needed (see [Backend Contract](#backend-contract)) |
| `backends/<yourname>/__init__.py` | `MANAGER_CLASS = YourManager` (relative import) |
| `config.yaml` (repo root, parent of `backend/`) | `memory.manager_class: <yourname>` + your knobs under `memory.backend_config` |
| `packages/harness/pyproject.toml` | **Only if the backend needs external libs**: declare the dependency; add `[tool.uv.sources]` for vendored source. Otherwise `uv sync` purges it (see [Common Pitfalls](#common-pitfalls)) |

See the docstring at the top of `noop/noop_manager.py` for the full 6-step walkthrough.

## Switch the Active Backend

Edit `config.yaml` (repo root) only:

```yaml
memory:
  manager_class: <name>        # mymem / noop / <yourname>
  backend_config: { ... }      # that backend's private config
```

Then **restart my-assistant** - the memory manager is a process-level singleton; a running process does not hot-reload config or backend code.

The default selector is `mymem`. Direct imports use
`my_assistant.agents.memory.backends.mymem.my_mem.MyMem` and
`my_assistant.agents.memory.backends.mymem.mymem.core.*`. The standalone engine
uses `MYMEM_DATA_DIR` or `~/.mymem` by default. The stored data format is
unchanged. The MyAssistant factory still defaults to
`.my_assistant/users/{user_id}/memory.json` and the adjacent agent facts.

## Backend Contract

### 1. The three-tier contract

`MemoryManager` is a pydantic `BaseModel` (not a bare ABC). Methods are tiered:

- **Tier 1 (abstract)** -- `add` + `get_context`: every backend MUST implement (write + read-inject are the backend's fundamental duties; missing one is caught at instantiation).
- **Tier 2 (management, with defaults)** -- `add_nowait` (delegates to `add`), `search` / `get_memory` / `clear_memory` / `import_memory` / `export_memory` / `delete_memory` (default `raise NotImplementedError`), `shutdown_flush` (default `True`). Override the ones your backend supports.
- **Tier 3 (optional hooks, with defaults)** -- `warm` (default `True`), `reload_memory` / `create_fact` / `delete_fact` / `update_fact` (default raise), `on_pre_compress` / `on_turn_start` (default no-op).

A new backend implements `from_config` + `add` + `get_context` and overrides only what it supports; the rest inherits defaults. Signatures must match (parameter names, keyword-only args). `noop` is the minimal reference.

### 2. Return shape

Each method follows the backend-neutral types and return contracts declared by `MemoryManager` in `manager.py`. Keep provider-specific records behind the manager contract.

### 3. Tier-3 hooks (contracted, no `hasattr` probing)

`create_fact` / `delete_fact` / `update_fact` / `reload_memory` / `warm` are tier-3 hooks ON the base contract (with defaults). Callers (client / tools / middleware) invoke them directly and catch `NotImplementedError` for unsupported backends -- no more `hasattr` probing.

- `create_fact` / `delete_fact` / `update_fact` - optional fact-management operations.
- `reload_memory` - optional explicit refresh operation.
- `warm` - one-time manager initialization hook (default `True` = nothing to warm).

Implement the ones your backend supports; the rest inherit the default raise.

### 4. Portability (the golden rule)

> [!IMPORTANT]
> A backend talks to the host through exactly **two channels**: (1) the ABC method arguments (`manager.py`), and (2) the `backend_config` dict. The **only** `from my_assistant` import allowed anywhere in your backend folder is the ABC contract line in `<name>_manager.py`:

```python
from my_assistant.agents.memory.manager import MemoryManager
```

Change that one line (and only that line) to port the backend to another agent. **Do not import my-assistant path helpers, config singletons, or models** - get `storage_path` and everything else from `backend_config`.

### 5. What the host provides

The factory (`manager.py::get_memory_manager`) resolves the backend class, injects `storage_path` into `backend_config`, then calls `cls.from_config(backend_config, mode=cfg.mode, **host_hooks)`. The host hooks (passed as `from_config` kwargs, NOT in `backend_config`):

- `backend_config["storage_path"]` (str) - a writable state dir (the host's `runtime_home` by default, or whatever `config.yaml` sets). **Use this as your storage root.**
- `callbacks` (`MemoryCallbacks` | None) - observability; `on_memory_llm_call` merges trace metadata before your LLM call (langfuse). Pass it to your LLM path; ignore if you don't trace.
- `should_keep_hidden_message` / `trace_context_manager` / `host_llm_factory` - other host hooks; consume in `from_config` if relevant.
- Plus whatever the user puts under `config.yaml::memory.backend_config` (your backend's own knobs).

Each backend's `from_config` consumes the hooks it needs (MyMem does; noop ignores them).

## Do Not Modify

These are backend-agnostic. Don't touch them when swapping backends (unless you're changing the **shared contract**, which affects every backend):

| File | Role |
|---|---|
| `packages/harness/my_assistant/agents/memory/manager.py` | ABC + factory + scanner |
| `packages/harness/my_assistant/agents/middlewares/memory_middleware.py` | `after_agent` -> `manager.add` |
| `packages/harness/my_assistant/agents/memory/summarization_hook.py` | summarization -> `manager.add_nowait` |
| `packages/harness/my_assistant/agents/lead_agent/prompt.py` | `_get_memory_context` -> `manager.get_context` |
| `packages/harness/my_assistant/config/memory_config.py` | shared 4 fields (`enabled` / `injection_enabled` / `manager_class` / `backend_config`) |

## Common Pitfalls

Lessons from integrating external backends:

1. **External deps must be declared in `pyproject.toml`.** A bare `uv pip install` is purged on the next `uv sync` / `langgraph dev`. Declare the dep (and `[tool.uv.sources]` for vendored source).
2. **Follow the manager contract.** Keep provider-specific records internal and return the documented types.
3. **Treat optional methods explicitly.** Unsupported methods raise `NotImplementedError`; implement only the operations the backend supports.
4. **Don't import `runtime_home`.** Read `storage_path` from `backend_config`. (The `noop` template shows the correct pattern; importing my-assistant path helpers breaks portability - contract #4.)
5. **Restart my-assistant after changes.** The manager is a process-level singleton; a running process does not hot-reload config or backend code.
6. **Cap `get_context` length yourself.** The host applies no token budget; the backend must truncate (MyMem has `max_injection_tokens`; noop does not).

## Honcho Backend

The optional `honcho/` backend is a remote-only HTTP adapter for user-model memory over Honcho (self-hosted or via api.honcho.dev). Middleware mode is the default; tool mode is also supported (search is implemented) and retains passive writes via `MemoryMiddleware` (`requires_passive_writes_in_tool_mode = True`) so Honcho's server-side deriver keeps building representations from every turn (no local LLM calls).

**Configuration** (under `memory.backend_config`):

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `base_url` | str | `http://localhost:8000` | Honcho instance URL (e.g., `http://localhost:8000` or `https://api.honcho.dev`) |
| `api_key` | str | optional | API key for hosted Honcho; required if `base_url` is `api.honcho.dev`. Can use `$HONCHO_API_KEY` env var syntax. Requires `allow_insecure_http: true` when using plain HTTP |
| `allow_insecure_http` | bool | false | Allow HTTP (non-HTTPS) connections; needed for localhost development with api_key |
| `timeout_seconds` | float | `10.0` | HTTP client timeout (seconds) for calls to Honcho — read/write/pool; see `connect_timeout_seconds` for the connect phase. Must be finite and `> 0` |
| `connect_timeout_seconds` | float | `3.0` | HTTP connect timeout (seconds) for establishing the connection to Honcho. Must be finite and `> 0` |
| `workspace_prefix` | str | `my_assistant-u-` | Prefix for isolated workspaces; each user gets one workspace named `{prefix}{sanitized_id}` |
| `workspace_overrides` | dict | `{}` | Map specific user ids to custom workspace names; overrides the prefix-based derivation. Values must be non-empty (parse error otherwise). Mapping several users to one workspace shares its search index across them (see Workspace Resolution) |
| `user_peer_overrides` | dict | `{}` | Map specific user ids to custom names for the user's own peer; overrides the stable-id derivation. Values must be non-empty (parse error otherwise) |
| `assistant_peer` | str | `my_assistant` | Default peer name for the assistant when storing messages |
| `message_char_limit` | int | `8000` | Character limit per message; longer messages are truncated. Must be `> 0` (zero empties the write; a negative value is a Python suffix slice, not a cap) |
| `max_injection_chars` | int | `6000` | Character limit for injected memory into the system prompt. Must be `> 0` |
| `failure_policy.read` | str | `fail_open` | Recall failure handling: `fail_open` (log and return empty) or `fail_closed` (rethrow) |

**Workspace Resolution**: Each MyAssistant user maps to one Honcho workspace. The workspace name is derived as: `workspace_overrides[user_id]` (if present) else `workspace_prefix + sanitized_id`, where `sanitized_id` is a collision-resistant hash suffix (sanitize[:48]-sha256[:8]). Missing user fails closed to no memory. The default derivation is isolated per user; a `workspace_overrides` entry that maps several users to one workspace deliberately shares that workspace's **search index** across them (`search` uses Honcho's workspace-scoped `/search`, which has no peer filter), while `get_context` / `get_memory` remain peer-scoped.

**Tool Mode**: While tool-mode memory tools are not fully supported, the backend implements `requires_passive_writes_in_tool_mode = True` to retain passive writes via MemoryMiddleware while also enabling memory search through the `memory_search` tool.

## Reference

- **Template**: `noop/` - minimal implementation with full docstrings; copy and go.
- **Contract + factory**: `packages/harness/my_assistant/agents/memory/manager.py` (`MemoryManager` base, `MemoryCallbacks`, `get_memory_manager` factory).
