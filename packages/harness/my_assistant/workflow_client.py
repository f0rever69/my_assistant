"""Trusted user controls for plan approval and task acceptance, never LLM tools."""

from __future__ import annotations

from my_assistant.agents.task_continuity import workflow
from my_assistant.config.workspace import Workspace
from my_assistant.runtime.checkpoint_state import CheckpointStateAccessor


class WorkflowClientMixin:
    def get_execution(self, thread_id):
        _, _, state = self._workspace_state(thread_id)
        return {"task": state.get("task"), "todos": state.get("todos", [])}

    def _execution_access(self, thread_id, workspace):
        self.check_thread_workspace(thread_id, workspace=workspace)
        accessor, config, state = self._workspace_state(thread_id)
        agent = getattr(self, "_agent", None)
        if agent is not None:
            snapshot = CheckpointStateAccessor.bind(agent, self._get_thread_checkpointer(), mode=self._checkpoint_channel_mode).get(config)
            if snapshot.next:
                raise PermissionError("Interrupted graph is pending; resume or abandon it before changing approval/acceptance")
        checkpoint = self._get_thread_checkpointer().get_tuple(config)
        if checkpoint is not None:
            versions = checkpoint.checkpoint.get("channel_versions", {})
            # Versions record historical routing too. Only unconsumed branch
            # values/pending writes mean a restarted graph is still running.
            routing = set(checkpoint.checkpoint.get("channel_values", {})) | {
                channel for _, channel, _ in (checkpoint.pending_writes or [])
                if channel.startswith("branch:to:")
            }
            if agent is None and any(k.startswith("branch:to:") and k != "branch:to:workspace_control" for k in routing):
                raise PermissionError("Restore the agent or use a new thread before changing a plan; pending routing is unknown")
            if any(not k.startswith(("__", "branch:")) and k not in accessor.graph.channels for k in versions):
                raise PermissionError("Restore this thread's effective agent schema first")
        if not (state.get("task") or {}).get("execution"):
            raise ValueError("No plan contract exists; enter /plan preview or submit a plan first")
        return accessor, config, state

    def approve_plan(self, thread_id, version: int, *, workspace=None):
        accessor, config, state = self._execution_access(thread_id, workspace)
        task = workflow.approve(state["task"], state.get("todos", []), version)
        accessor.update(config, {"task": task}, as_node="workspace_control")
        return {"task": task, "todos": state.get("todos", [])}

    def check_task_acceptance(self, thread_id, *, workspace=None, confirm=False):
        accessor, config, state = self._execution_access(thread_id, workspace)
        task = state["task"]
        if task["execution"]["phase"] not in {"delivered", "completed"}:
            raise ValueError("Task has not been delivered; approval or running is not acceptance")
        task = workflow.check_delivery(task, Workspace.from_context(workspace), thread_id, human_confirm=confirm, todos=state.get("todos", []))
        accessor.update(config, {"task": task}, as_node="workspace_control")
        return {"task": task, "todos": state.get("todos", [])}

    def resume_execution(self, thread_id, *, workspace=None):
        accessor, config, state = self._execution_access(thread_id, workspace)
        task = workflow.resume(state["task"], state.get("todos", []))
        accessor.update(config, {"task": task}, as_node="workspace_control")
        return {"task": task, "todos": state.get("todos", [])}
