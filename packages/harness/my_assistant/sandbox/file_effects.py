"""Runtime-owned file-operation location facts, independent of model claims."""

from pathlib import Path

from my_assistant.config.workspace import Workspace

FILE_EFFECT_KEY = "my_assistant_file_effect"
DELIVERY_MESSAGE_PREFIX = "workspace-delivery-"
_FIELDS = frozenset({"operation", "path", "storage_scope", "sync_status", "host_path"})


def valid_file_effect(value: object) -> bool:
    return isinstance(value, dict) and set(value) == _FIELDS and all(
        isinstance(item, str) and len(item) <= 4096 for item in value.values()
    ) and value.get("storage_scope") in {
        "sandbox_workspace_copy", "host_workspace", "host_outputs", "sandbox_execution", "host_execution", "unknown",
    } and value.get("sync_status") in {
        "pending_apply", "host_updated", "host_output_updated", "host_update_not_proven", "unverified",
    }


def copy_mode(config) -> bool:
    sandbox = getattr(config, "sandbox", None)
    return "AioSandboxProvider" in str(getattr(sandbox, "use", "")) and getattr(sandbox, "isolate_workspace", False) is True


def delivery_context(runtime, state: dict, config):
    from my_assistant.runtime.user_context import resolve_runtime_user_id

    if not copy_mode(config):
        return None
    context = getattr(runtime, "context", None) or {}
    workspace = Workspace.from_context(context.get("workspace") or (state.get("thread_data") or {}).get("workspace_policy"))
    thread = context.get("thread_id")
    if workspace is None or not isinstance(thread, str) or not thread:
        return None
    binding = state.get("workspace_binding")
    if binding and (binding.get("workspace_id") != workspace.id or binding.get("root_fingerprint") != workspace.root_fingerprint):
        raise PermissionError("Workspace delivery binding mismatch")
    return workspace, thread, resolve_runtime_user_id(runtime)


def describe_file_effect(request, config) -> dict | None:
    tool = getattr(request, "tool", None)
    if (getattr(tool, "metadata", None) or {}).get("my_assistant_mcp"):
        return None
    function = getattr(tool, "func", None) or getattr(tool, "coroutine", None)
    if getattr(function, "__module__", "") != "my_assistant.sandbox.tools":
        return None
    call = request.tool_call
    operation = call.get("name")
    if operation not in {"write_file", "str_replace", "delete_file", "bash"}:
        return None
    runtime, state = request.runtime, request.state or {}
    from my_assistant.sandbox.tools import is_local_sandbox

    local = is_local_sandbox(runtime)
    context = getattr(runtime, "context", None) or {}
    data = state.get("thread_data") or {}
    workspace = Workspace.from_context(context.get("workspace") or data.get("workspace_policy"))
    path = str((call.get("args") or {}).get("path") or "")
    result = {"operation": operation, "path": path, "storage_scope": "unknown", "sync_status": "unverified", "host_path": ""}
    if operation == "bash":
        result.update(path="/mnt/user-data/workspace", storage_scope="host_execution" if local else "sandbox_execution", sync_status="host_update_not_proven")
    elif workspace is not None and (path == "/mnt/user-data/workspace" or path.startswith("/mnt/user-data/workspace/")):
        relative = path.removeprefix("/mnt/user-data/workspace").lstrip("/")
        result["host_path"] = str(workspace.check(relative, write=True))
        if local:
            result.update(storage_scope="host_workspace", sync_status="host_updated")
        elif copy_mode(config):
            result.update(storage_scope="sandbox_workspace_copy", sync_status="pending_apply")
    elif path.startswith("/mnt/user-data/outputs/") and data.get("outputs_path") and (local or copy_mode(config)):
        root = Workspace.from_path(data["outputs_path"])
        result.update(host_path=str(root.resolve(Path(path.removeprefix("/mnt/user-data/outputs/")))), storage_scope="host_outputs", sync_status="host_output_updated")
    return result
