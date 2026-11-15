import logging
from datetime import UTC, datetime
from typing import NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

from my_assistant.agents.thread_state import ThreadDataState
from my_assistant.config.paths import Paths, get_paths
from my_assistant.runtime.user_context import resolve_runtime_user_id

logger = logging.getLogger(__name__)


class ThreadDataMiddlewareState(AgentState):
    """Compatible with the `ThreadState` schema."""

    thread_data: NotRequired[ThreadDataState | None]
    workspace_binding: NotRequired[dict | None]


class ThreadDataMiddleware(AgentMiddleware[ThreadDataMiddlewareState]):
    """Create thread data directories for each thread execution.

    Creates the following directory structure:
    - {base_dir}/threads/{thread_id}/user-data/workspace
    - {base_dir}/threads/{thread_id}/user-data/uploads
    - {base_dir}/threads/{thread_id}/user-data/outputs

    Lifecycle Management:
    - With lazy_init=True (default): Only compute paths, directories created on-demand
    - With lazy_init=False: Eagerly create directories in before_agent()
    """

    state_schema = ThreadDataMiddlewareState

    def __init__(self, base_dir: str | None = None, lazy_init: bool = True):
        """Initialize the middleware.

        Args:
            base_dir: Base directory for thread data. Defaults to Paths resolution.
            lazy_init: If True, defer directory creation until needed.
                      If False, create directories eagerly in before_agent().
                      Default is True for optimal performance.
        """
        super().__init__()
        self._paths = Paths(base_dir) if base_dir else get_paths()
        self._lazy_init = lazy_init

    def release_policy_parameters(self) -> dict[str, object]:
        return {"workspace_binding_version": 1, "revocation_before_each_tool": True, "lazy_init": self._lazy_init}

    def _get_thread_paths(self, thread_id: str, user_id: str | None = None) -> dict[str, str]:
        """Get the paths for a thread's data directories.

        Args:
            thread_id: The thread ID.
            user_id: Optional user ID for per-user path isolation.

        Returns:
            Dictionary with workspace_path, uploads_path, and outputs_path.
        """
        return {
            "workspace_path": str(self._paths.sandbox_work_dir(thread_id, user_id=user_id)),
            "uploads_path": str(self._paths.sandbox_uploads_dir(thread_id, user_id=user_id)),
            "outputs_path": str(self._paths.sandbox_outputs_dir(thread_id, user_id=user_id)),
        }

    def _create_thread_directories(self, thread_id: str, user_id: str | None = None) -> dict[str, str]:
        """Create the thread data directories.

        Args:
            thread_id: The thread ID.
            user_id: Optional user ID for per-user path isolation.

        Returns:
            Dictionary with the created directory paths.
        """
        self._paths.ensure_thread_dirs(thread_id, user_id=user_id)
        return self._get_thread_paths(thread_id, user_id=user_id)

    @override
    def before_agent(self, state: ThreadDataMiddlewareState, runtime: Runtime) -> dict | None:
        context = runtime.context or {}
        thread_id = context.get("thread_id")
        if thread_id is None:
            config = get_config()
            thread_id = config.get("configurable", {}).get("thread_id")

        if thread_id is None:
            raise ValueError("Thread ID is required in runtime context or config.configurable")

        user_id = resolve_runtime_user_id(runtime)
        from my_assistant.config.workspace import Workspace

        workspace_context = context.get("workspace")
        workspace = Workspace.from_context(workspace_context)
        if workspace_context is not None and workspace is None:
            raise PermissionError("Workspace policy is invalid")

        from my_assistant.workspaces import get_workspace_registry

        binding = get_workspace_registry(user_id).check(thread_id, state, workspace, bind=True)

        if workspace is not None:
            paths = {
                "workspace_path": str(workspace.root),
                **workspace.ensure_runtime_dirs(thread_id),
            }
        elif self._lazy_init:
            # Lazy initialization: only compute paths, don't create directories
            paths = self._get_thread_paths(thread_id, user_id=user_id)
        else:
            # Eager initialization: create directories immediately
            paths = self._create_thread_directories(thread_id, user_id=user_id)
            logger.debug("Created thread data directories for thread %s", thread_id)

        messages = list(state.get("messages", []))
        last_message = messages[-1] if messages else None

        if last_message and isinstance(last_message, HumanMessage):
            messages[-1] = HumanMessage(
                content=last_message.content,
                id=last_message.id,
                name=last_message.name or "user-input",
                additional_kwargs={**last_message.additional_kwargs, "run_id": context.get("run_id"), "timestamp": datetime.now(UTC).isoformat()},
            )

        return {
            "workspace_binding": binding,
            **({"task": {**state["task"], "workspace_binding": binding}} if state.get("task") else {}),
            "thread_data": {
                **paths,
                "workspace_policy": workspace.to_context() if workspace is not None else None,
            },
            "messages": messages,
        }

    def _check_tool_workspace(self, request) -> None:
        """Recheck ownership/revocation at execution, including long-lived runs."""
        from my_assistant.config.workspace import Workspace
        from my_assistant.workspaces import get_workspace_registry

        runtime = request.runtime
        context = runtime.context or {}
        workspace = Workspace.from_context(context.get("workspace"))
        if context.get("workspace") is not None and workspace is None:
            raise PermissionError("Workspace is unavailable; no tool was executed")
        thread_id = context.get("thread_id")
        if thread_id:
            get_workspace_registry(resolve_runtime_user_id(runtime)).check(thread_id, runtime.state, workspace)

    def wrap_tool_call(self, request, handler):
        try:
            self._check_tool_workspace(request)
        except (PermissionError, LookupError, OSError) as exc:
            return ToolMessage(content=f"Error: {exc}", tool_call_id=request.tool_call["id"], status="error")
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        try:
            self._check_tool_workspace(request)
        except (PermissionError, LookupError, OSError) as exc:
            return ToolMessage(content=f"Error: {exc}", tool_call_id=request.tool_call["id"], status="error")
        return await handler(request)
