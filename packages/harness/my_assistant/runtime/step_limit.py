"""Shared user-facing explanation for a bounded LangGraph run."""

from __future__ import annotations


def format_step_limit_error(limit: object = None) -> str:
    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        bound = f" ({limit})"
    else:
        bound = ""
    return (
        f"Agent stopped at the configured graph step limit{bound} before producing a final answer. "
        "Any intermediate messages and tool results already streamed remain visible. "
        "Narrow the request or raise recursion_limit to continue."
    )
