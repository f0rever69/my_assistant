"""The Textual application — a terminal workbench over the embedded harness.

The app keeps a single immutable :class:`ViewState` and re-renders it through the
pure renderers. Agent runs execute on a worker *thread* (``MyAssistantClient.stream``
is a synchronous generator); each yielded action is marshalled back onto the UI
thread via ``call_from_thread`` and folded into the reducer.
"""

from __future__ import annotations

import copy
import shlex
import subprocess
import uuid
from functools import partial
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from my_assistant.runtime.goal import parse_goal_command

from .clipboard import copy_native_text
from .command_registry import format_command_help
from .input_history import InputHistory
from .local_review import open_document
from .render import render_header, render_progress, render_status, render_transcript
from .review_screens import FileComparisonScreen, WorkspaceReviewScreen
from .runtime import project_progress, stream_actions
from .text_export import selectable_rich_text, transcript_text
from .text_widgets import ConflictReviewScreen, CopyableStatic, TextViewScreen
from .theme import SYMBOLS, THEME
from .view_state import (
    ClearRows,
    RunEnded,
    RunStarted,
    SystemMessage,
    ThreadTitle,
    UserSubmitted,
    initial_state,
    reduce,
)
from .widgets.composer import ComposerInput

_HELP_KEYS = "Keys: Enter send · PgUp/PgDn scroll · Select text + Ctrl+C copy; otherwise interrupt/quit · F8 or /copy text view · Ctrl+L redraw"
_HELP_TEXT = (
    f"{format_command_help()}\n"
    'Memory: /memory list · /memory search <text> · /memory forget <fact-id> · '
    '/memory correct <fact-id> "replacement"\n'
    'Plan: /plan on|off|preview|show · /plan approve <version> · /accept [show|check|confirm]\n'
    'Workspace: /workspace refresh [--dry-run] · /workspace diff [path] · /workspace apply <path> <sha256>\n'
    'Review: /artifacts [file] opens local outputs · /workspace review [path] reviews one file or a batch\n'
    'Conflicts: /workspace conflicts [path] opens version-choice dialog; reviewed tokens are carried automatically\n'
    f"{_HELP_KEYS}"
)


_TRANSPARENT_CSS = """
Screen,
#header,
#scroll,
#progress,
#status,
#palette,
#composer,
SelectScreen #dialog,
SelectScreen OptionList {
    background: ansi_default;
}
"""


class SelectScreen(ModalScreen):
    """A centered modal that returns the id of the chosen option (or None)."""

    BINDINGS: ClassVar = [Binding("escape", "cancel", "Close")]

    def __init__(self, title: str, options: list[tuple[str, str]]) -> None:
        super().__init__()
        self._title = title
        self._options = options

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self._title, id="dialog-title")
            yield OptionList(
                *[Option(label, id=oid) for oid, label in self._options],
                id="dialog-list",
            )

    def on_mount(self) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option_id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class MyAssistantTUI(App):
    CSS = f"""
    Screen {{
        background: {THEME.bg};
        color: {THEME.text};
    }}
    #header {{
        height: auto;
        max-height: 5;
        padding: 0 1;
        background: {THEME.panel};
    }}
    #scroll {{
        height: 1fr;
        padding: 1 2;
        background: {THEME.bg};
        scrollbar-size-vertical: 1;
    }}
    #transcript {{
        width: 100%;
        height: auto;
    }}
    #status {{
        height: 1;
        padding: 0 1;
        background: {THEME.panel};
        color: {THEME.muted};
    }}
    #progress {{
        height: auto;
        max-height: 10;
        padding: 0 1;
        background: {THEME.panel};
    }}
    #palette {{
        height: auto;
        max-height: 10;
        margin: 0 1;
        padding: 0 1;
        background: {THEME.panel};
        border: round {THEME.border};
        display: none;
    }}
    #palette.open {{
        display: block;
    }}
    #composer {{
        height: 3;
        margin: 0 1 1 1;
        border: round {THEME.border};
        background: {THEME.panel};
    }}
    #composer:focus {{
        border: round {THEME.primary};
    }}
    SelectScreen {{
        align: center middle;
    }}
    SelectScreen #dialog {{
        width: 72;
        max-width: 90%;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: {THEME.panel};
        border: round {THEME.primary};
    }}
    SelectScreen #dialog-title {{
        color: {THEME.primary};
        text-style: bold;
        padding: 0 0 1 0;
    }}
    SelectScreen OptionList {{
        background: {THEME.panel};
        border: none;
        height: auto;
        max-height: 20;
    }}
    SelectScreen OptionList > .option-list--option-highlighted {{
        background: {THEME.primary};
        color: {THEME.bg};
    }}
    """

    BINDINGS: ClassVar = [
        Binding("ctrl+c", "interrupt", "Interrupt / Quit", priority=True, show=True),
        Binding("ctrl+l", "redraw", "Redraw", show=False),
        Binding("f8", "text_view", "Copyable text", show=False),
        Binding("ctrl+u", "clear_composer", "Clear input", show=False),
        # Up/Down drive the palette when it's open, otherwise input history.
        # Tab/Enter/Esc only act when the palette is open. check_action gates all
        # of these so they never steal keys from a modal overlay or the composer.
        Binding("down", "nav_down", show=False, priority=True),
        Binding("up", "nav_up", show=False, priority=True),
        Binding("tab", "palette_complete", show=False, priority=True),
        Binding("escape", "escape", show=False, priority=True),
        Binding("enter", "palette_accept", show=False, priority=True),
        Binding("pageup", "transcript_page_up", show=False, priority=True),
        Binding("pagedown", "transcript_page_down", show=False, priority=True),
    ]

    def __init__(self, session, plan) -> None:
        transparent = bool(getattr(plan, "transparent", False))
        if transparent:
            self.CSS = f"{self.CSS}\n{_TRANSPARENT_CSS}"
        super().__init__(ansi_color=True if transparent else None)
        self.session = session
        self.plan = plan
        self.state = initial_state()
        self._conv_thread_id: str | None = None
        self._model = ""
        self._skill_names: list[str] = []
        self._skills = 0
        self._spinner_idx = 0
        self._streaming = False
        self._cancelled = False
        self._run_token: str | None = None
        self._skills_meta: list[dict] = []
        self._model_override: str | None = None
        self._palette_open = False
        self._palette_items: list = []
        self._palette_index = 0
        self._history = InputHistory()
        self._transcript_dirty = False
        self._transcript_scroll_pending = False
        self._workspace_context: dict | None = None
        self._workspace_error: str | None = None
        self._thread_workspace_binding: dict | None = None
        self._workspace_blocked = False
        self._plan_enabled = bool(getattr(plan, "plan_mode", False))
        self._preview_next = False

    # ----- composition --------------------------------------------------- #

    def compose(self) -> ComposeResult:
        yield CopyableStatic(id="header")
        with VerticalScroll(id="progress"):
            yield CopyableStatic(id="progress-content")
        with VerticalScroll(id="scroll"):
            yield CopyableStatic(id="transcript")
        yield CopyableStatic(id="status")
        yield CopyableStatic(id="palette")
        yield ComposerInput(
            placeholder="Message my_assistant…   ( / for commands )", id="composer"
        )

    def on_mount(self) -> None:
        self._load_session_info()
        self._refresh_all()
        self.set_interval(0.1, self._tick_spinner)
        self.set_interval(0.06, self._flush_transcript)  # coalesce streaming re-renders
        self.query_one("#composer", Input).focus()
        if self._workspace_error:
            self._dispatch(SystemMessage(self._workspace_error, tone="error"))
        if self._conv_thread_id:
            self._check_workspace_access(report=True)
            self._refresh_header()
        if self._workspace_context is None and not self._conv_thread_id and not self._workspace_error:
            from my_assistant.workspaces import get_workspace_registry

            if any(row["available"] for row in get_workspace_registry().list()):
                self._open_workspace_picker(initial_message=getattr(self.plan, "message", None))
                return
        if self.plan and getattr(self.plan, "message", None):
            self._send_to_agent(self.plan.message)

    # ----- session info -------------------------------------------------- #

    def _load_session_info(self) -> None:
        from .cli import _workspace_context

        try:
            self._workspace_context = _workspace_context(self.plan) if self.plan else None
        except Exception as exc:  # noqa: BLE001 - keep the TUI open to select another workspace
            self._workspace_error = str(exc)
        self._conv_thread_id = (
            self.session.resolve_thread(self.plan) if self.plan else None
        )
        if self._conv_thread_id is not None:
            self._restore_thread_state(self._conv_thread_id)
            try:
                self._thread_workspace_binding = self.session.client.get_thread_workspace(
                    self._conv_thread_id
                ).get("workspace_binding")
            except Exception:  # noqa: BLE001 - legacy threads may not have binding metadata
                self._thread_workspace_binding = None
        client = self.session.client
        try:
            models = client.list_models().get("models", [])
            self._model = next(
                (
                    m.get("display_name") or m.get("name")
                    for m in models
                    if m.get("name")
                ),
                "",
            )
        except Exception:  # noqa: BLE001 - header is best-effort
            self._model = ""
        try:
            skills = client.list_skills(enabled_only=True).get("skills", [])
            self._skills_meta = [s for s in skills if s.get("name")]
            self._skill_names = [s["name"] for s in self._skills_meta]
            self._skills = len(self._skill_names)
        except Exception:  # noqa: BLE001
            self._skills_meta = []
            self._skill_names = []
            self._skills = 0

    # ----- input --------------------------------------------------------- #

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        self._close_palette()
        if not text:
            return
        self._history.add(text)
        self._handle_submit(text)

    def on_input_changed(self, event: Input.Changed) -> None:
        value = event.value
        if value.startswith("/") and " " not in value:
            from .command_registry import build_registry, filter_commands

            items = filter_commands(build_registry(self._skills_meta), value[1:])
            # The candidate set changed, so the previous highlight index is stale —
            # reset to the top rather than clamping to a now-different command.
            self._palette_index = 0
            self._open_palette(items)
        else:
            self._close_palette()

    # ----- slash command palette ----------------------------------------- #

    def check_action(self, action: str, parameters):
        if action == "interrupt" and (
            isinstance(self.screen, (TextViewScreen, WorkspaceReviewScreen))
            or getattr(self.screen, "selections", {})
            or isinstance(self.focused, Input) and self.focused.selected_text
        ):
            # Let the TextArea/Screen copy binding receive Ctrl+C.
            return None
        custom = {
            "nav_up",
            "nav_down",
            "palette_complete",
            "palette_accept",
            "escape",
            "transcript_page_up",
            "transcript_page_down",
            "text_view",
        }
        if action in custom:
            # A modal overlay (e.g. the model/thread picker) is on top — never
            # intercept its keys; let the overlay handle them natively.
            if len(self.screen_stack) > 1:
                return None
            # History navigation, transcript paging, Tab, and Esc are always
            # consumed. Enter falls through to the Input when the palette is
            # closed so it submits normally.
            if action in {
                "nav_up",
                "nav_down",
                "palette_complete",
                "escape",
                "transcript_page_up",
                "transcript_page_down",
                "text_view",
            }:
                return True
            return True if self._palette_open else None
        return True

    def action_nav_up(self) -> None:
        if self._palette_open:
            self.action_palette_up()
        else:
            self._history_move(
                self._history.up(self.query_one("#composer", Input).value)
            )

    def action_nav_down(self) -> None:
        if self._palette_open:
            self.action_palette_down()
        else:
            self._history_move(self._history.down())

    def _history_move(self, value: str) -> None:
        composer = self.query_one("#composer", Input)
        composer.value = value
        composer.cursor_position = len(value)

    def _open_palette(self, items: list) -> None:
        if not items:
            self._close_palette()
            return
        self._palette_items = items
        self._palette_index = min(self._palette_index, len(items) - 1)
        self._palette_open = True
        palette = self.query_one("#palette", Static)
        palette.add_class("open")
        self._render_palette()

    def _close_palette(self) -> None:
        if not self._palette_open and not self._palette_items:
            return
        self._palette_open = False
        self._palette_items = []
        self._palette_index = 0
        self.query_one("#palette", Static).remove_class("open")

    def _render_palette(self) -> None:
        from .render import render_palette

        self.query_one("#palette", Static).update(
            render_palette(self._palette_items, self._palette_index)
        )

    def _current_palette_item(self):
        if 0 <= self._palette_index < len(self._palette_items):
            return self._palette_items[self._palette_index]
        return None

    def action_palette_down(self) -> None:
        if self._palette_items:
            self._palette_index = min(
                self._palette_index + 1, len(self._palette_items) - 1
            )
            self._render_palette()

    def action_palette_up(self) -> None:
        if self._palette_items:
            self._palette_index = max(self._palette_index - 1, 0)
            self._render_palette()

    def action_palette_complete(self) -> None:
        # When the palette is open, Tab completes the highlighted command.
        # When it's closed, Tab is a no-op here (consumed) so focus stays in the
        # composer instead of moving to the scroll region.
        if self._palette_open:
            self._fill_from_palette()

    def action_palette_accept(self) -> None:
        item = self._current_palette_item()
        if item is None:
            return
        if getattr(item, "category", "") == "skill":
            # Skills need a task argument; fill and let the user keep typing.
            self._fill_from_palette()
            return
        self._close_palette()
        self.query_one("#composer", Input).value = ""
        self._handle_submit(f"/{item.name}")

    def _fill_from_palette(self) -> None:
        item = self._current_palette_item()
        if item is None:
            return
        composer = self.query_one("#composer", Input)
        composer.value = f"/{item.name} "
        composer.cursor_position = len(composer.value)
        self._close_palette()

    def _handle_submit(self, text: str) -> None:
        from .command_registry import resolve

        res = resolve(text, skills=self._skill_names)
        if res.kind == "builtin":
            self._handle_builtin(res.name, res.args)
            return
        if res.kind == "unknown":
            self._dispatch(
                SystemMessage(f"Unknown command /{res.name}. Try /help.", tone="error")
            )
            return
        # plain message or skill activation (/skill task) both go to the agent,
        # which applies skill-activation semantics on the raw text.
        self._send_to_agent(text)

    def _dispatch_still_working(self) -> None:
        self._dispatch(
            SystemMessage(
                "Still working — wait for the current run to finish.", tone="info"
            )
        )

    def _handle_builtin(self, name: str, args: str) -> None:
        if name == "quit":
            # Mirror action_interrupt (Ctrl+C): an active run must be interrupted
            # before we tear the app down. Otherwise the worker thread is left
            # running against an app that no longer exists — its next
            # call_from_thread fails silently and the in-flight turn (plus any
            # post-run persistence, e.g. thread title) is quietly abandoned.
            if self._streaming:
                self._interrupt_run()
            self.exit()
        elif name == "help":
            self._dispatch(SystemMessage(_HELP_TEXT))
        elif name == "copy":
            self._show_text_view(args.strip() or "all")
        elif name == "new":
            if self._streaming:
                self._dispatch_still_working()
                return
            self._conv_thread_id = None
            self._thread_workspace_binding = None
            self._workspace_blocked = False
            self.state = initial_state()
            self._dispatch(SystemMessage("Started a new thread."))
        elif name == "clear":
            if self._streaming:
                self._dispatch_still_working()
                return
            self._dispatch(ClearRows())
        elif name == "model":
            self._open_model_picker()
        elif name == "resume":
            self._resume_thread(args)
        elif name == "workspace":
            self._handle_workspace(args)
        elif name == "writes":
            self._handle_file_writes(args)
        elif name == "goal":
            self._handle_goal(args)
        elif name == "task":
            self._handle_task(args)
        elif name == "plan":
            self._handle_plan(args)
        elif name == "accept":
            self._handle_accept(args)
        elif name == "askkb":
            self._handle_knowledge_question(args)
        elif name == "skills":
            self._show_skills()
        elif name == "mcp":
            self._show_mcp()
        elif name == "memory":
            self._show_memory(args)
        elif name == "usage":
            self._show_usage()
        elif name == "config":
            self._show_config()
        elif name == "tools":
            self._dispatch(
                SystemMessage(
                    "Tools are listed in the agent's runtime; use /mcp for MCP servers."
                )
            )
        elif name == "uploads":
            self._show_uploads()
        elif name == "artifacts":
            self._open_delivery_outputs(args)
        elif name == "details":
            self._dispatch(
                SystemMessage(
                    "Verbose activity is always shown in this build.", tone="info"
                )
            )
        else:
            self._dispatch(SystemMessage(f"/{name} is not available yet.", tone="info"))

    def _open_delivery_outputs(self, args: str) -> None:
        if self._streaming:
            self._dispatch_still_working()
            return
        thread = self._conv_thread_id
        workspace = copy.deepcopy(self._workspace_context)
        if thread is None:
            self._dispatch(SystemMessage("请先打开一个会话。"))
            return
        client = self.session.client

        def open_output(name):
            if name is None:
                return
            if self._streaming or thread != self._conv_thread_id or workspace != self._workspace_context:
                self._dispatch(SystemMessage("会话状态已改变，请重新打开输出列表。", tone="error"))
                return
            try:
                path = client.resolve_delivery_output(thread, workspace, name)
            except (OSError, ValueError) as exc:
                self._dispatch(SystemMessage(f"无法定位输出文件：{exc}", tone="error"))
                return
            try:
                open_document(path)
            except (OSError, ValueError) as exc:
                self._dispatch(SystemMessage(f"{exc}\n文件路径：{path}", tone="error"))
            else:
                self._dispatch(SystemMessage(f"已请求打开本地输出文件：{path}"))

        if args.strip():
            open_output(args.strip().strip('"'))
            return
        try:
            outputs = client.list_delivery_outputs(thread, workspace)
        except (OSError, ValueError) as exc:
            self._dispatch(SystemMessage(f"无法读取输出目录：{exc}", tone="error"))
            return
        if not outputs:
            self._dispatch(SystemMessage("当前会话还没有本地输出文件。"))
            return
        options = [(item["path"], f"{item['path']} · {item['size']:,} 字节") for item in outputs]
        self.push_screen(SelectScreen("选择要打开的本地输出文件", options), open_output)

    def _handle_file_writes(self, args: str) -> None:
        if self._streaming:
            self._dispatch_still_working()
            return
        if not self._conv_thread_id:
            self._dispatch(
                SystemMessage(
                    "Start or resume a thread before managing file writes.",
                    tone="error",
                )
            )
            return
        parts = args.split()
        action = parts[0].lower() if parts else "list"
        workspace = self._workspace_context
        try:
            if action in {"list", ""}:
                operations = self.session.client.list_file_write_operations(
                    self._conv_thread_id, workspace=workspace
                )
                if not operations:
                    self._dispatch(
                        SystemMessage(
                            "No file write operations recorded for this thread."
                        )
                    )
                    return
                lines = [
                    f"{row['status']}  {row['operation_id']}  {row['display_path']}"
                    + (f"  ({row['error']})" if row.get("error") else "")
                    for row in operations
                ]
                self._dispatch(SystemMessage("File writes:\n" + "\n".join(lines)))
                return
            if action == "continue":
                self._send_to_agent(
                    "Continue recovery of interrupted file writes.",
                    resume_file_writes=True,
                )
                return
            if action in {"rollback", "abandon"} and len(parts) == 2:
                operation_id = parts[1]
                if action == "rollback":
                    self.session.client.rollback_file_write_operation(
                        self._conv_thread_id, operation_id, workspace=workspace
                    )
                    self._dispatch(
                        SystemMessage(
                            f"Restored the file before-image for {operation_id}; reconciling its tool result into the checkpoint."
                        )
                    )
                else:
                    self.session.client.abandon_file_write_operation(
                        self._conv_thread_id, operation_id, workspace=workspace
                    )
                    self._dispatch(
                        SystemMessage(
                            f"Abandoned {operation_id}; reconciling its tool result into the checkpoint."
                        )
                    )
                self._send_to_agent(
                    f"Reconcile file write {operation_id} after user {action}."
                )
                return
            if action == "undo" and len(parts) == 1:
                undone = self.session.client.undo_latest_file_write(
                    self._conv_thread_id, workspace=workspace
                )
                self._dispatch(
                    SystemMessage(
                        f"Undid the latest write to {undone['display_path']} ({undone['operation_id']})."
                    )
                )
                self._send_to_agent(
                    f"Reconcile file write {undone['operation_id']} after user rollback."
                )
                return
            self._dispatch(
                SystemMessage(
                    "Usage: /writes [list|continue|undo|rollback <operation_id>|abandon <operation_id>]",
                    tone="error",
                )
            )
        except Exception as exc:  # noqa: BLE001 - surface storage and conflict errors in the UI
            self._dispatch(
                SystemMessage(
                    f"File write operation failed: {str(exc) or exc.__class__.__name__}",
                    tone="error",
                )
            )

    # ----- overlays + info commands -------------------------------------- #

    def _open_model_picker(self) -> None:
        try:
            models = self.session.client.list_models().get("models", [])
        except Exception:  # noqa: BLE001
            models = []
        options = [
            (m["name"], (m.get("display_name") or m["name"]))
            for m in models
            if m.get("name")
        ]
        if not options:
            self._dispatch(SystemMessage("No models configured.", tone="error"))
            return

        def on_choice(choice: str | None) -> None:
            if choice:
                self._model_override = choice
                self._model = choice
                self._dispatch(SystemMessage(f"Model set to {choice}."))
                self._refresh_header()

        self.push_screen(SelectScreen("Select model", options), on_choice)

    def _open_resume_picker(self) -> None:
        from my_assistant.workspaces import get_workspace_registry

        try:
            threads = self.session.recent_threads(limit=20)
        except Exception:  # noqa: BLE001
            threads = []
        registry = get_workspace_registry()
        catalog = {row["id"]: row for row in registry.list()}
        options: list[tuple[str, str]] = []
        for thread in threads:
            tid = thread.get("thread_id")
            if not tid:
                continue
            title = thread.get("title") or "untitled"
            binding = registry.thread_binding(tid)
            workspace_label = "legacy/unbound"
            if binding:
                workspace_label = "thread-local"
                if binding.get("workspace_id"):
                    record = catalog.get(binding["workspace_id"], {})
                    workspace_label = record.get("name", binding["workspace_id"][-8:])
                    if not record.get("available"):
                        workspace_label = f"unavailable:{workspace_label}"
            title = f"{title} [{workspace_label}]"
            options.append((tid, f"{title}   ·   {tid[:8]}"))
        if not options:
            self._dispatch(SystemMessage("No saved threads yet."))
            return

        def on_choice(choice: str | None) -> None:
            if choice:
                self._switch_to_thread(choice)

        self.push_screen(SelectScreen("Resume thread", options), on_choice)

    def _resume_thread(self, ref: str) -> None:
        """/resume [id-or-title]: switch to a thread, or open the picker if blank."""
        ref = ref.strip()
        if not ref:
            self._open_resume_picker()
            return
        self._switch_to_thread(self.session.resolve_ref(ref))

    def _switch_to_thread(self, thread_id: str) -> None:
        if self._streaming:
            self._dispatch_still_working()
            return
        self._conv_thread_id = thread_id
        restored = self._restore_thread_state(thread_id)
        if restored:
            self._dispatch(SystemMessage(f"Resumed thread {thread_id[:8]}."))
        self._check_workspace_access(report=True)
        self._refresh_header()

    def _check_workspace_access(self, *, report: bool = False) -> bool:
        """Read-only preflight: restoring history never rebinds a thread."""
        checker = getattr(self.session.client, "check_thread_workspace", None)
        if checker is None:
            return True  # lightweight UI test/integration clients
        try:
            if self._workspace_error:
                raise PermissionError(self._workspace_error)
            if self._workspace_context:
                from my_assistant.config.workspace import Workspace
                from my_assistant.workspaces import get_workspace_registry

                current = Workspace.from_context(self._workspace_context)
                if current is None:
                    raise PermissionError("Workspace root is missing or its fingerprint changed")
                get_workspace_registry().validate(current)
            if self._conv_thread_id:
                self._thread_workspace_binding = self.session.client.get_thread_workspace(
                    self._conv_thread_id
                ).get("workspace_binding")
                checker(self._conv_thread_id, workspace=self._workspace_context)
            self._workspace_blocked = False
            return True
        except (PermissionError, LookupError, ValueError, OSError) as exc:
            self._workspace_blocked = True
            if report:
                self._dispatch(SystemMessage(
                    f"Workspace blocked: {exc}\nUse /workspace return, or /workspace migrate <target-id> confirm for a task-only new thread.",
                    tone="error",
                ))
            return False

    def _select_workspace(self, ref: str | None, *, keep_thread: bool = False) -> None:
        from my_assistant.workspaces import get_workspace_registry

        context = get_workspace_registry().resolve(ref).to_context() if ref else None
        if keep_thread and self._conv_thread_id:
            self.session.client.check_thread_workspace(self._conv_thread_id, workspace=context)
        self._workspace_context = context
        self._workspace_error = None
        self._workspace_blocked = False
        if not keep_thread:
            self._conv_thread_id = None
            self._thread_workspace_binding = None
            self.state = initial_state()
        self.session.client.reset_agent()
        self._dispatch(SystemMessage(
            f"Workspace: {context['name']} ({context['id']})." if context else "Thread-local storage selected."
        ))
        self._refresh_header()

    def _open_workspace_picker(self, *, initial_message: str | None = None) -> None:
        from my_assistant.workspaces import get_workspace_registry

        options = [(row["id"], f"{row['name']} · {row['root_path']} · {row['id']}") for row in get_workspace_registry().list() if row["available"]]
        options.append(("__thread_local__", "Thread-local storage (no project Workspace)"))

        def on_choice(choice):
            if choice:
                try:
                    self._select_workspace(None if choice == "__thread_local__" else choice)
                    if initial_message:
                        self._send_to_agent(initial_message)
                except Exception as exc:  # noqa: BLE001 - command error belongs in the transcript
                    self._dispatch(SystemMessage(str(exc), tone="error"))

        self.push_screen(SelectScreen("Select Workspace", options), on_choice)

    def _handle_workspace(self, args: str) -> None:
        if self._streaming:
            self._dispatch_still_working()
            return
        import json

        from my_assistant.workspaces import get_workspace_registry

        registry = get_workspace_registry()
        try:
            parts = [item.strip('"') for item in shlex.split(args, posix=False)]
            action = parts[0].lower() if parts else "select"
            if action == "select" and len(parts) <= 1:
                self._open_workspace_picker()
                return
            if action in {"list", "show"}:
                rows = registry.list() if action == "list" else ([registry.record(self._workspace_context["id"])] if self._workspace_context else [])
                self._dispatch(SystemMessage(json.dumps(rows, ensure_ascii=False, indent=2) if rows else "No project Workspace selected. Use /workspace add <path>."))
                return
            if action == "add" and len(parts) in {2, 3}:
                workspace = registry.register(parts[1], name=parts[2] if len(parts) == 3 else None)
                self._select_workspace(workspace.id)
                return
            if action == "use" and len(parts) == 2:
                self._select_workspace(None if parts[1] == "local" else parts[1])
                return
            if action == "return" and len(parts) == 1 and self._conv_thread_id:
                binding = self.session.client.get_thread_workspace(self._conv_thread_id)["workspace_binding"]
                if binding is None:
                    raise ValueError("No saved Workspace binding; explicitly migrate this legacy task")
                self._select_workspace(binding.get("workspace_id"), keep_thread=True)
                return
            if action == "policy":
                command = args.split(maxsplit=1)
                if len(command) != 2:
                    raise ValueError("Usage: /workspace policy <id|name> {JSON}")
                payload = command[1]
                json_start = payload.find("{")
                if json_start < 0:
                    raise ValueError("Workspace policy must be followed by a JSON object")
                ref = payload[:json_start].strip().strip('"')
                patch = json.loads(payload[json_start:])
                if not isinstance(patch, dict):
                    raise ValueError("Workspace policy must be a JSON object")
                workspace = registry.update(ref, **patch)
                if self._workspace_context and workspace.id == self._workspace_context["id"]:
                    self._workspace_context = workspace.to_context()
                    self._workspace_error = None
                self.session.client.reset_agent()
                self._dispatch(SystemMessage("Workspace policy updated:\n" + json.dumps(workspace.to_context(), ensure_ascii=False, indent=2)))
                self._refresh_header()
                return
            if action == "remove" and len(parts) == 3 and parts[2] == "confirm":
                record = registry.record(parts[1])
                registry.delete(record["id"])
                if self._workspace_context and self._workspace_context["id"] == record["id"]:
                    self._workspace_error = "Workspace registration deleted; file actions unavailable"
                self._dispatch(SystemMessage("Registration removed. Files, tasks and conversations retained."))
                self._refresh_header()
                return
            if action == "migrate" and len(parts) == 3 and parts[2] == "confirm" and self._conv_thread_id:
                workspace = registry.resolve(parts[1])
                result = self.session.client.migrate_task_workspace(self._conv_thread_id, workspace.to_context())
                self._workspace_context = workspace.to_context()
                self._workspace_error = None
                self._switch_to_thread(result["thread_id"])
                self._dispatch(SystemMessage("Task migrated to a NEW paused thread. Only objective, constraints and pending steps copied; no history, file operations or index. Review with /task show before /task continue."))
                return
            if action == "export" and len(parts) == 2:
                if not self._workspace_context:
                    raise ValueError("Select the source Workspace first")
                result = self.session.client.export_workspace_file(self._workspace_context, parts[1])
                self._dispatch(SystemMessage("Explicit snapshot export:\n" + json.dumps(result, ensure_ascii=False, indent=2)))
                return
            if action in {"diff", "apply", "refresh", "conflicts", "resolve", "review"}:
                if not self._workspace_context or not self._conv_thread_id:
                    raise ValueError("Open a Workspace thread before reviewing Docker changes")
                client = self.session.client
                thread = self._conv_thread_id
                workspace = self._workspace_context
                if action == "review" and len(parts) == 2:
                    self.push_screen(FileComparisonScreen(client.compare_workspace_file(thread, workspace, parts[1])))
                elif action == "review" and len(parts) == 1:
                    self._open_batch_review(thread, copy.deepcopy(workspace))
                elif action == "conflicts" and len(parts) == 1:
                    conflicts = client.list_workspace_conflicts(thread, workspace)
                    self._dispatch(SystemMessage(json.dumps(conflicts, ensure_ascii=False, indent=2) if conflicts else "No Workspace conflicts."))
                    if len(conflicts) == 1:
                        self._open_conflict_review(conflicts[0]["path"])
                    elif conflicts:
                        def choose_file(path):
                            if path is not None and thread == self._conv_thread_id and workspace == self._workspace_context:
                                self._open_conflict_review(path)
                        self.push_screen(SelectScreen("选择要处理的冲突文件", [(item["path"], item["path"]) for item in conflicts]), choose_file)
                elif action == "conflicts" and len(parts) == 2:
                    review = client.review_workspace_conflict(thread, workspace, parts[1])
                    self._dispatch(SystemMessage(f"正在审阅 {parts[1]} 的本地与沙箱版本，请在弹窗中选择处理方式。"))
                    self._open_conflict_review(parts[1], review)
                elif action == "resolve" and len(parts) == 4:
                    result = client.resolve_workspace_conflict(thread, workspace, parts[1], parts[2], parts[3])
                    label = "已保留本地版本并同步到沙箱。" if parts[2] == "local" else "已采用沙箱版本，本地已更新。"
                    self._dispatch(SystemMessage(label + "双方原版本已备份。\n" + json.dumps(result, ensure_ascii=False, indent=2)))
                elif action == "refresh" and (len(parts) == 1 or parts[1:] == ["--dry-run"]):
                    preview = len(parts) == 2
                    result = client.refresh_workspace_copy(thread, workspace, dry_run=preview)
                    headline = (
                        "Refresh blocked by conflicts; no files changed."
                        if result["conflicts"] else
                        "Host-to-sandbox refresh preview; no files changed."
                        if preview else "Host changes refreshed into the sandbox copy."
                    )
                    self._dispatch(SystemMessage(headline + "\n" + json.dumps(result, ensure_ascii=False, indent=2)))
                elif action == "diff" and len(parts) == 1:
                    changes = client.diff_workspace_copy(thread, workspace)
                    self._dispatch(SystemMessage(json.dumps(changes, ensure_ascii=False, indent=2) if changes else "No Docker Workspace changes."))
                elif action == "diff" and len(parts) == 2:
                    review = client.review_workspace_copy(thread, workspace, parts[1])
                    self._dispatch(SystemMessage(f"{review['diff']}\nSHA-256 {review['sha256']}"))
                elif action == "apply" and len(parts) == 3:
                    path = client.apply_workspace_copy(thread, workspace, parts[1], parts[2])
                    self._dispatch(SystemMessage(f"本地已更新（已审核写回）：{path}。"))
                else:
                    raise ValueError("Usage: /workspace review [path], refresh [--dry-run], diff [path], apply <path> <sha256>, conflicts [path], resolve <path> local|sandbox <token>")
                return
            if action == "exports" and len(parts) == 1:
                self._dispatch(SystemMessage(json.dumps(self.session.client.list_workspace_exports(), ensure_ascii=False, indent=2)))
                return
            if action == "forget-export" and len(parts) == 3 and parts[2] == "confirm":
                self.session.client.forget_workspace_export(parts[1])
                self._dispatch(SystemMessage("Export revoked for future references. Previously attached snapshots remain in their threads."))
                return
            if action == "reference" and len(parts) == 2:
                if not self._workspace_context:
                    raise ValueError("Select the target Workspace first")
                thread_id = self._conv_thread_id or str(uuid.uuid4())
                result = self.session.client.reference_workspace_export(thread_id, self._workspace_context, parts[1])
                self._conv_thread_id = thread_id
                self._dispatch(SystemMessage("Snapshot attached; tell the assistant to read this path. No live source access:\n" + json.dumps(result, ensure_ascii=False, indent=2)))
                self._refresh_header()
                return
            self._dispatch(SystemMessage('Usage: /workspace [list|show|add <path> [name]|use <id|name|local>|return|policy <id> {JSON}|remove <id> confirm|migrate <id> confirm|review [path]|refresh [--dry-run]|diff [path]|apply <path> <sha256>|conflicts [path]|resolve <path> local|sandbox <token>|export <relative-path>|exports|reference <export-id>|forget-export <id> confirm]', tone="error"))
        except Exception as exc:  # noqa: BLE001 - registry/parse failures must not terminate TUI
            self._dispatch(SystemMessage(f"Workspace: {exc}", tone="error"))

    def _open_batch_review(self, thread: str, workspace: dict) -> None:
        client = self.session.client
        changes = client.list_workspace_review(thread, workspace)
        if not changes:
            self._dispatch(SystemMessage("当前没有待审阅的沙箱变更。"))
            return
        task = client.get_task(thread).get("task") or {}
        checks = (task.get("execution") or {}).get("checks") or []

        def scope_is_current():
            return not self._streaming and thread == self._conv_thread_id and workspace == self._workspace_context

        def compare(change):
            if not scope_is_current():
                self.notify("会话状态已改变，请重新打开审阅。", severity="error")
                return
            try:
                comparison = client.compare_workspace_file(thread, workspace, change.path, expected_token=change.token)
            except Exception as exc:  # noqa: BLE001 - keep the chooser open on review errors
                self.notify(str(exc), severity="error", timeout=5)
                return
            self.push_screen(FileComparisonScreen(comparison))

        def apply_selection(selected):
            if not selected:
                return
            if not scope_is_current():
                self._dispatch(SystemMessage("会话状态已改变，请重新审阅。", tone="error"))
                return
            try:
                result = client.apply_workspace_review(thread, workspace, selected)
            except Exception as exc:  # noqa: BLE001 - preflight failures belong in the transcript
                self._dispatch(SystemMessage(f"批量预检查未通过，未开始写回。\n{exc}", tone="error"))
                return
            lines = [f"原工作区已完成 {len(result['applied'])} 项变更。"]
            for item in result["applied"]:
                label = "已删除" if item["status"] == "host_deleted" else "已写回"
                lines.append(f"- {item['path']} · {label} · 备份 {item['backup_path']}")
            if result["error"]:
                lines.append(f"{result['failed_path']} 写回中断，请重新核验该文件。\n{result['error']}")
                if result["not_attempted"]:
                    lines.append(f"尚未尝试其余 {len(result['not_attempted'])} 项。")
            self._dispatch(SystemMessage("\n".join(lines), tone="error" if result["error"] else "info"))

        self.push_screen(WorkspaceReviewScreen(changes, checks, compare), apply_selection)

    def _open_conflict_review(self, path: str, review: dict | None = None) -> None:
        import json

        thread = self._conv_thread_id
        workspace = copy.deepcopy(self._workspace_context)
        if self._streaming:
            self._dispatch_still_working()
            return
        if not thread or not workspace:
            return
        try:
            review = review or self.session.client.review_workspace_conflict(thread, workspace, path)
        except Exception as exc:  # noqa: BLE001 - scope or files may change while a picker is open
            self._dispatch(SystemMessage(f"无法打开冲突审阅：{exc}", tone="error"))
            return

        def choose_version(choice):
            if choice is None:
                return
            if self._streaming:
                self._dispatch_still_working()
                return
            if thread != self._conv_thread_id or workspace != self._workspace_context:
                self._dispatch(SystemMessage("线程或工作区已改变，请重新查看冲突。", tone="error"))
                return
            try:
                result = self.session.client.resolve_workspace_conflict(thread, workspace, path, choice, review["resolution_token"])
            except Exception as exc:  # noqa: BLE001 - keep user-controlled resolution failures visible
                self._dispatch(SystemMessage(f"未应用选择：{exc}\n请重新 /workspace conflicts 查看最新版本。", tone="error"))
                return
            label = "已保留本地版本并同步到沙箱。" if choice == "local" else "已采用沙箱版本，本地已更新。"
            self._dispatch(SystemMessage(label + "双方原版本已备份。\n" + json.dumps(result, ensure_ascii=False, indent=2)))

        self.push_screen(ConflictReviewScreen(review), choose_version)

    def copy_to_clipboard(self, text: str) -> None:
        # Keep Textual's clipboard/terminal fallback (also used for paste), and
        # use the native Windows clipboard when terminal OSC 52 is unsupported.
        super().copy_to_clipboard(text)
        try:
            native = copy_native_text(text)
        except (OSError, subprocess.SubprocessError):
            native = False
        self.notify("已复制到系统剪贴板。" if native else "已向终端发送复制请求；如未生效，可在文本视图中重试。", timeout=3)

    def _show_text_view(self, mode: str = "all") -> None:
        if mode not in {"all", "last"}:
            self._dispatch(SystemMessage("Usage: /copy [all|last]", tone="error"))
            return
        text = transcript_text(self.state, last=(mode == "last"))
        if mode == "all":
            progress = selectable_rich_text(render_progress(self.state, enabled=self._plan_enabled, preview=self._preview_next), self.console, 120).plain
            status = render_status(self.state, model=self._model, thread_label=self._thread_label()).plain
            header = f"线程：{self._conv_thread_id or '新线程'}\n模型：{self._model or '默认'}\n工作区：{(self._workspace_context or {}).get('root_path', '线程存储')}\n"
            text = header + "\n" + progress + "\n" + text + "\n\n" + status
        self.push_screen(TextViewScreen(text, "全部终端文字" if mode == "all" else "最近一条消息"))

    def action_text_view(self) -> None:
        self._show_text_view()

    def _restore_thread_state(self, thread_id: str) -> bool:
        """Load a saved transcript without letting one bad checkpoint stop the TUI."""
        try:
            self.state = self.session.load_view_state(thread_id)
            try:
                task = self.session.client.get_task(thread_id).get("task")
                if task and task.get("status") in {"active", "paused", "blocked", "delivered"}:
                    self._dispatch(
                        SystemMessage(
                            self._format_task(task)
                            + self._task_hint(task)
                        )
                    )
            except Exception:  # noqa: BLE001,S110 - task hint is best-effort during transcript recovery
                pass
            return True
        except Exception as exc:  # noqa: BLE001 - recovery must keep the TUI usable
            self.state = reduce(
                initial_state(),
                SystemMessage(
                    f"Could not restore thread: {str(exc) or exc.__class__.__name__}",
                    tone="error",
                ),
            )
            return False

    @staticmethod
    def _task_hint(task: dict) -> str:
        phase = (task.get("execution") or {}).get("phase")
        if phase == "awaiting_approval":
            return f"\nReview /plan show, then /plan approve {task['execution']['plan_version']} and /task continue."
        if phase == "delivered":
            return "\nDelivered, awaiting acceptance. Use /accept show or /accept check; /accept confirm accepts the results."
        return "\nUse /task continue to resume, /task show for details, or /task abandon to stop."

    @staticmethod
    def _format_task(task: dict) -> str:
        lines = [f"Unfinished task [{task.get('status')}]: {task.get('objective')}"]
        if task.get("next_action"):
            lines.append(f"Next: {task['next_action']}")
        if task.get("pending_steps"):
            lines.append(f"Pending steps: {len(task['pending_steps'])}")
        if task.get("modified_files"):
            lines.append("Modified files: " + ", ".join(task["modified_files"][-8:]))
        return "\n".join(lines)

    def _handle_task(self, args: str) -> None:
        if self._streaming:
            self._dispatch_still_working()
            return
        parts = args.split(maxsplit=1)
        action = parts[0].lower() if parts else "show"
        value = parts[1].strip() if len(parts) > 1 else ""
        if action == "start":
            if not value:
                self._dispatch(
                    SystemMessage("Usage: /task start <objective>", tone="error")
                )
                return
            if self._conv_thread_id is None:
                self._conv_thread_id = str(uuid.uuid4())
                self._refresh_header()
            try:
                task = self.session.client.set_task(
                    self._conv_thread_id, value, workspace=self._workspace_context
                )["task"]
                self._dispatch(
                    SystemMessage("Task started:\n" + self._format_task(task))
                )
            except Exception as exc:  # noqa: BLE001 - surface checkpoint errors in the TUI
                self._dispatch(
                    SystemMessage(
                        f"Could not start task: {str(exc) or exc.__class__.__name__}",
                        tone="error",
                    )
                )
            return
        if not self._conv_thread_id:
            self._dispatch(
                SystemMessage(
                    "Start or resume a thread before managing a task.", tone="error"
                )
            )
            return
        try:
            task = self.session.client.get_task(self._conv_thread_id).get("task")
            if action == "show":
                self._dispatch(
                    SystemMessage(
                        self._format_task(task)
                        if task
                        else "No structured task for this thread."
                    )
                )
            elif action in {"pause", "complete", "abandon"}:
                if not task:
                    self._dispatch(SystemMessage("No structured task for this thread."))
                    return
                status = {
                    "pause": "paused",
                    "complete": "completed",
                    "abandon": "abandoned",
                }[action]
                updated = self.session.client.update_task(
                    self._conv_thread_id, {}, status=status
                )["task"]
                if updated.get("execution"):
                    self._dispatch(project_progress(self.session.client.get_execution(self._conv_thread_id)))
                self._dispatch(
                    SystemMessage(f"Task {status}: {updated.get('objective')}")
                )
            elif action == "continue":
                if not self._check_workspace_access(report=True):
                    return
                if not task:
                    self._dispatch(
                        SystemMessage(
                            "No structured task for this thread.", tone="error"
                        )
                    )
                    return
                if task.get("status") in {"completed", "abandoned"}:
                    self._dispatch(
                        SystemMessage(
                            f"Task is {task.get('status')} and cannot be resumed.",
                            tone="error",
                        )
                    )
                    return
                workspace = self._workspace_context
                operations = self.session.client.list_file_write_operations(
                    self._conv_thread_id, workspace=workspace
                )
                conflicts = [
                    row for row in operations if row.get("status") == "CONFLICT"
                ]
                if conflicts:
                    reasons = "; ".join(
                        f"{row.get('display_path')}: {row.get('error') or 'file changed'}"
                        for row in conflicts
                    )
                    self.session.client.update_task(
                        self._conv_thread_id,
                        {
                            "add_failed_attempts": [
                                "File recovery conflict: " + reasons
                            ],
                            "next_action": "Resolve the file conflict, then review the task.",
                        },
                        status="blocked",
                    )
                    self._dispatch(
                        SystemMessage(
                            "Task blocked by file transaction conflict:\n" + reasons,
                            tone="error",
                        )
                    )
                    return
                prepared = any(row.get("status") == "PREPARED" for row in operations)
                if task.get("execution"):
                    result = self.session.client.resume_execution(self._conv_thread_id, workspace=workspace)
                    self._dispatch(project_progress(result))
                else:
                    self.session.client.update_task(self._conv_thread_id, {}, status="active")
                self._send_to_agent(
                    "Continue the saved task from its checkpointed next action.",
                    resume_file_writes=prepared,
                    resume_task=True,
                )
            else:
                self._dispatch(
                    SystemMessage(
                        "Usage: /task [show|start <objective>|continue|pause|complete|abandon]",
                        tone="error",
                    )
                )
        except Exception as exc:  # noqa: BLE001 - surface task/file recovery errors in the TUI
            self._dispatch(
                SystemMessage(
                    f"Task operation failed: {str(exc) or exc.__class__.__name__}",
                    tone="error",
                )
            )

    def _handle_goal(self, args: str) -> None:
        if self._streaming:
            self._dispatch_still_working()
            return
        command = parse_goal_command(args)

        if command.kind == "status":
            if not self._conv_thread_id:
                self._dispatch(SystemMessage("No active goal."))
                return
            try:
                goal = self.session.client.get_goal(self._conv_thread_id).get("goal")
            except Exception:  # noqa: BLE001
                self._dispatch(SystemMessage("Could not read goal.", tone="error"))
                return
            if not goal:
                self._dispatch(SystemMessage("No active goal."))
                return
            self._dispatch(SystemMessage(f"Goal: {goal.get('objective')}"))
            return

        if command.kind == "clear":
            if self._conv_thread_id:
                try:
                    self.session.client.clear_goal(self._conv_thread_id)
                except Exception:  # noqa: BLE001
                    self._dispatch(SystemMessage("Could not clear goal.", tone="error"))
                    return
            self._dispatch(SystemMessage("Goal cleared."))
            return

        if self._conv_thread_id is None:
            self._conv_thread_id = str(uuid.uuid4())
            self._refresh_header()
        try:
            if not self._check_workspace_access(report=True):
                return
            goal = self.session.client.set_goal(
                self._conv_thread_id, command.objective, workspace=self._workspace_context
            ).get("goal")
        except Exception:  # noqa: BLE001
            self._dispatch(SystemMessage("Could not set goal.", tone="error"))
            return
        self._dispatch(
            SystemMessage(
                f"Goal set: {goal.get('objective') if goal else command.objective}"
            )
        )

    def _show_skills(self) -> None:
        names = ", ".join(self._skill_names) or "none"
        self._dispatch(SystemMessage(f"Enabled skills ({self._skills}): {names}"))

    def _show_mcp(self) -> None:
        try:
            servers = self.session.client.get_mcp_config().get("mcp_servers", {})
        except Exception:  # noqa: BLE001
            self._dispatch(SystemMessage("Could not read MCP config.", tone="error"))
            return
        if not servers:
            self._dispatch(SystemMessage("No MCP servers configured."))
            return
        lines = [
            f"{name}: {'on' if cfg.get('enabled') else 'off'}"
            for name, cfg in servers.items()
        ]
        self._dispatch(SystemMessage("MCP servers — " + "  ·  ".join(lines)))

    def _show_memory(self, args: str = "") -> None:
        try:
            data = self.session.client.get_memory()
        except Exception:  # noqa: BLE001
            self._dispatch(SystemMessage("Could not read memory.", tone="error"))
            return
        facts = data.get("facts", []) if isinstance(data, dict) else []
        parts = args.split(maxsplit=1)
        action = parts[0].lower() if parts else "list"
        value = parts[1] if len(parts) > 1 else ""
        if action == "search":
            if not value:
                self._dispatch(SystemMessage("Usage: /memory search <text>", tone="error"))
                return
            facts = [fact for fact in facts if value.casefold() in str(fact.get("content", "")).casefold()]
        elif action == "forget":
            if not value:
                self._dispatch(SystemMessage("Usage: /memory forget <fact-id>", tone="error"))
                return
            try:
                self.session.client.delete_memory_fact(value.strip())
            except KeyError:
                self._dispatch(SystemMessage(f"Memory fact not found: {value.strip()}", tone="error"))
                return
            except Exception as exc:  # noqa: BLE001
                self._dispatch(SystemMessage(f"Could not forget memory: {exc}", tone="error"))
                return
            self._dispatch(SystemMessage(f"Forgot memory fact {value.strip()}."))
            return
        elif action == "correct":
            try:
                fact_id, replacement = value.split(maxsplit=1)
                parsed = shlex.split(replacement)
                if len(parsed) != 1:
                    raise ValueError
                content = parsed[0]
            except ValueError:
                self._dispatch(SystemMessage('Usage: /memory correct <fact-id> "corrected fact"', tone="error"))
                return
            try:
                self.session.client.update_memory_fact(
                    fact_id,
                    content=content,
                    category="correction",
                    confidence=1.0,
                    source_thread_id=self._conv_thread_id,
                )
            except KeyError:
                self._dispatch(SystemMessage(f"Memory fact not found: {fact_id}", tone="error"))
                return
            except Exception as exc:  # noqa: BLE001
                self._dispatch(SystemMessage(f"Could not correct memory: {exc}", tone="error"))
                return
            self._dispatch(SystemMessage(f"Corrected memory fact {fact_id}; the previous value is no longer active."))
            return
        elif action != "list":
            self._dispatch(SystemMessage('Usage: /memory [list|search <text>|forget <fact-id>|correct <fact-id> "text"]', tone="error"))
            return
        if not facts:
            self._dispatch(SystemMessage("No matching long-term memory facts."))
            return
        lines = []
        for fact in facts[:30]:
            source = fact.get("sourceThreadId") or fact.get("source") or "unknown"
            evidence = fact.get("evidenceType", "legacy/unknown")
            updated = fact.get("updatedAt") or fact.get("createdAt") or "unknown"
            ttl = fact.get("ttlDays")
            expiry = " · expired" if fact.get("expired") else (f" · TTL {ttl}d" if ttl else "")
            lines.append(f"{fact.get('id', '?')} [{evidence}] {fact.get('content', '')}\n  source: {source} · updated: {updated}{expiry}")
        self._dispatch(SystemMessage("Long-term memory\n" + "\n".join(lines)))

    def _show_usage(self) -> None:
        usage = self.state.usage or {}
        if not usage:
            self._dispatch(SystemMessage("No token usage recorded yet."))
            return
        parts = ", ".join(f"{k}={v}" for k, v in usage.items())
        self._dispatch(SystemMessage(f"Token usage — {parts}"))

    def _show_config(self) -> None:
        import os

        self._dispatch(
            SystemMessage(f"cwd: {os.getcwd()}   model: {self._model or 'default'}")
        )

    def _show_uploads(self) -> None:
        if not self._conv_thread_id:
            self._dispatch(SystemMessage("Start a thread before listing uploads."))
            return
        try:
            uploads = self.session.client.list_uploads(self._conv_thread_id).get(
                "files", []
            )
        except Exception:  # noqa: BLE001
            self._dispatch(SystemMessage("Could not list uploads.", tone="error"))
            return
        if not uploads:
            self._dispatch(SystemMessage("No uploads in this thread."))
            return
        names = ", ".join(f.get("filename", "?") for f in uploads)
        self._dispatch(SystemMessage(f"Uploads ({len(uploads)}): {names}"))

    # ----- agent run ----------------------------------------------------- #

    def _handle_plan(self, args: str) -> None:
        if self._streaming:
            self._dispatch_still_working()
            return
        parts = args.split()
        action = parts[0] if parts else "show"
        if action in {"on", "off", "preview"} and len(parts) == 1:
            self._plan_enabled = action != "off"
            self._preview_next = action == "preview"
            self._dispatch(SystemMessage(f"Plan tracking {action}. Preview applies to the next request; /plan off never revokes an existing confirmation gate."))
            return
        try:
            if not self._conv_thread_id:
                raise ValueError("No current task; use /plan preview, then describe your task.")
            if action == "approve" and len(parts) == 2:
                if not self._check_workspace_access(report=True):
                    return
                result = self.session.client.approve_plan(self._conv_thread_id, int(parts[1]), workspace=self._workspace_context)
                self._dispatch(project_progress(result))
                self._dispatch(SystemMessage("Plan version and exact actions approved. Use /task continue to execute; deletion and high-risk gates still apply."))
            elif action == "show" and len(parts) <= 1:
                import json

                result = self.session.client.get_execution(self._conv_thread_id)
                self._dispatch(project_progress(result))
                self._dispatch(SystemMessage(json.dumps(result, ensure_ascii=False, indent=2)))
            else:
                raise ValueError("Usage: /plan [on|off|preview|show|approve <version>]")
        except Exception as exc:  # noqa: BLE001 - local command failures stay visible
            self._dispatch(SystemMessage(f"Plan: {exc}", tone="error"))

    def _handle_accept(self, args: str) -> None:
        if self._streaming:
            self._dispatch_still_working()
            return
        try:
            action = args.strip() or "show"
            if action not in {"show", "check", "confirm"} or not self._conv_thread_id:
                raise ValueError("Usage: /accept [show|check|confirm], in a delivered task thread")
            if action == "show":
                result = self.session.client.get_execution(self._conv_thread_id)
            else:
                if not self._check_workspace_access(report=True):
                    return
                result = self.session.client.check_task_acceptance(self._conv_thread_id, workspace=self._workspace_context, confirm=action == "confirm")
            self._dispatch(project_progress(result))
            import json

            task = result.get("task") or {}
            self._dispatch(SystemMessage(json.dumps({"criteria": task.get("acceptance_criteria", []), "execution": task.get("execution")}, ensure_ascii=False, indent=2)))
        except Exception as exc:  # noqa: BLE001 - acceptance failure never terminates TUI
            self._dispatch(SystemMessage(f"Acceptance: {exc}", tone="error"))

    def _handle_knowledge_question(self, question: str) -> None:
        if self._streaming:
            self._dispatch_still_working()
            return
        if not question.strip():
            self._dispatch(SystemMessage("Usage: /askkb <question>", tone="error"))
            return
        workspace_id = (self._workspace_context or {}).get("id")
        if not workspace_id or not self._check_workspace_access(report=True):
            if not workspace_id:
                self._dispatch(SystemMessage("Select a registered Workspace with /workspace.", tone="error"))
            return
        self._cancelled = False
        self._run_token = str(uuid.uuid4())
        self._dispatch(UserSubmitted(question))
        self._on_action(RunStarted())
        self.run_worker(
            partial(self._knowledge_worker, question, workspace_id, self._run_token),
            thread=True, exclusive=True, group="agent",
        )

    def _knowledge_worker(self, question: str, workspace_id: str, run_token: str) -> None:
        from my_assistant.rag.local import open_index, open_service

        try:
            index = open_index(workspace_id=workspace_id)
            service, config = open_service(index)
            answer = service.ask(question, mode=config.output_mode)
            result = SystemMessage(
                f"{answer.render()}\ntrace_id: {answer.trace_id}",
                tone="error" if answer.error else "info",
            )
        except Exception as exc:
            detail = str(exc) if isinstance(exc, (LookupError, PermissionError, ValueError)) else "Knowledge base request failed."
            result = SystemMessage(detail, tone="error")
        if run_token == self._run_token:
            self.call_from_thread(self._on_run_action, run_token, result)
            self.call_from_thread(self._on_run_action, run_token, RunEnded())

    def _send_to_agent(
        self, text: str, *, resume_file_writes: bool = False, resume_task: bool = False
    ) -> None:
        if self._streaming:
            self._dispatch_still_working()
            return
        if self._workspace_error:
            self._dispatch(SystemMessage(self._workspace_error, tone="error"))
            return
        if not self._check_workspace_access():
            self._dispatch(SystemMessage(
                "This thread cannot run in the selected Workspace. Use /workspace return or explicitly migrate its unfinished task.",
                tone="error",
            ))
            return
        if self._conv_thread_id is None:
            self._conv_thread_id = str(uuid.uuid4())
        self._cancelled = False
        self._run_token = str(uuid.uuid4())
        self._streaming = True
        if not resume_task:
            self._dispatch(UserSubmitted(text))
        self.run_worker(
            partial(
                self._stream_worker,
                text,
                self._conv_thread_id,
                resume_file_writes,
                resume_task,
                copy.deepcopy(self._workspace_context),
                self._run_token,
                self._plan_enabled,
                self._preview_next,
            ),
            thread=True,
            exclusive=True,
            group="agent",
        )
        self._preview_next = False

    def _stream_worker(
        self,
        text: str,
        thread_id: str,
        resume_file_writes: bool = False,
        resume_task: bool = False,
        workspace_context: dict | None = None,
        run_token: str | None = None,
        plan_enabled: bool = False,
        preview: bool = False,
    ) -> None:
        if run_token != self._run_token:
            return
        kwargs: dict = {}
        kwargs["plan_mode"] = plan_enabled
        if preview:
            kwargs["plan_preview"] = True
        if workspace_context is not None:
            kwargs["workspace"] = workspace_context
        if resume_file_writes:
            kwargs["resume_file_writes"] = True
        if resume_task:
            kwargs["resume_task"] = True
        if self._model_override:
            kwargs["model_name"] = self._model_override

        writer = getattr(self.session, "writer", None)
        if writer is not None:
            # Persist terminal-session metadata alongside the checkpoint by writing a
            # threads_meta row under the local default user (best-effort, no-op on
            # memory backends). Done on this worker thread to keep the UI responsive.
            writer.ensure_created(
                thread_id,
                assistant_id="lead-agent",
                metadata={"source": "tui", "workspace_id": (workspace_context or {}).get("id")},
            )

        latest_title: str | None = None
        for action in stream_actions(
            self.session.client, text, thread_id=thread_id, **kwargs
        ):
            if self._cancelled or run_token != self._run_token:
                break
            if isinstance(action, ThreadTitle):
                latest_title = action.title
            self.call_from_thread(self._on_run_action, run_token, action)

        # Only persist a title for a run that completed normally — an interrupted
        # run may only have emitted the title middleware's first, truncated guess.
        if writer is not None and latest_title and not self._cancelled and run_token == self._run_token:
            writer.set_title(thread_id, latest_title)

    def _on_run_action(self, run_token, action) -> None:
        if run_token == self._run_token:
            self._on_action(action)

    def _on_action(self, action) -> None:
        self.state = reduce(self.state, action)
        if isinstance(action, RunStarted):
            self._streaming = True
            self._transcript_dirty = True
        elif isinstance(action, RunEnded):
            self._streaming = False
            # Flush now so the finished message snaps to its Markdown rendering.
            self._transcript_dirty = False
            self._refresh_transcript()
        else:
            # Coalesce rapid streaming deltas; _flush_transcript renders them.
            self._transcript_dirty = True
        self._refresh_status()

    def _flush_transcript(self) -> None:
        if self._transcript_dirty:
            self._transcript_dirty = False
            self._refresh_transcript()

    # ----- key actions --------------------------------------------------- #

    def action_interrupt(self) -> None:
        if self._streaming:
            self._interrupt_run()
        else:
            self.exit()

    def action_escape(self) -> None:
        if self.screen.selections:
            self.screen.clear_selection()
        elif self._palette_open:
            self._close_palette()
        elif self._streaming:
            self._interrupt_run()

    def _interrupt_run(self) -> None:
        self._cancelled = True
        self._run_token = None
        self.workers.cancel_group(self, "agent")
        self._streaming = False
        self.state = reduce(self.state, RunEnded())
        self._dispatch(SystemMessage("Interrupted.", tone="info"))

    def action_redraw(self) -> None:
        self.refresh(layout=True)
        self._refresh_all()

    def action_clear_composer(self) -> None:
        self.query_one("#composer", Input).value = ""

    def action_transcript_page_up(self) -> None:
        scroll = self.query_one("#scroll", VerticalScroll)
        self._transcript_scroll_pending = True
        scroll.scroll_page_up(animate=False, on_complete=self._finish_transcript_scroll)

    def action_transcript_page_down(self) -> None:
        scroll = self.query_one("#scroll", VerticalScroll)
        self._transcript_scroll_pending = True
        scroll.scroll_page_down(
            animate=False, on_complete=self._finish_transcript_scroll
        )

    def _finish_transcript_scroll(self) -> None:
        self._transcript_scroll_pending = False

    # ----- rendering ----------------------------------------------------- #

    def _dispatch(self, action) -> None:
        self.state = reduce(self.state, action)
        self._refresh_transcript()
        self._refresh_status()

    def _tick_spinner(self) -> None:
        if self._streaming:
            self._spinner_idx = (self._spinner_idx + 1) % len(SYMBOLS["spinner"])
            self._refresh_status()

    def _thread_label(self) -> str:
        if not self._conv_thread_id:
            return "new thread"
        return f"thread {self._conv_thread_id[:8]}"

    def _refresh_all(self) -> None:
        self._refresh_header()
        self._refresh_transcript()
        self._refresh_status()

    def _refresh_header(self) -> None:
        import os

        self.query_one("#header", Static).update(
            render_header(
                model=self._model,
                thread_label=self._thread_label(),
                cwd=os.getcwd(),
                skills=self._skills,
                workspace=self._workspace_context,
                workspace_error=self._workspace_error or ("binding mismatch" if self._workspace_blocked else None),
                thread_workspace_id=(self._thread_workspace_binding or {}).get("workspace_id"),
            )
        )

    def _refresh_transcript(self) -> None:
        scroll = self.query_one("#scroll", VerticalScroll)
        follow_transcript = scroll.is_vertical_scroll_end
        reading_position = scroll.scroll_y
        self.query_one("#transcript", Static).update(render_transcript(self.state))
        if self._transcript_scroll_pending:
            # Let the scheduled page movement win if output arrives between the
            # key press and Textual's next refresh cycle.
            return
        if follow_transcript:
            scroll.scroll_end(animate=False)
        else:
            # Static.update() can change layout before the next refresh. Restore
            # the absolute reading position after that layout has been applied.
            scroll.scroll_to(y=reading_position, animate=False)

    def _refresh_status(self) -> None:
        self.query_one("#progress-content", Static).update(render_progress(self.state, enabled=self._plan_enabled, preview=self._preview_next))
        spinner = SYMBOLS["spinner"][self._spinner_idx] if self._streaming else ""
        self.query_one("#status", Static).update(
            render_status(
                self.state,
                model=self._model,
                thread_label=self._thread_label(),
                spinner=spinner,
            )
        )


def run_tui(plan) -> int:
    """Construct the embedded session and run the app. Returns a process exit code."""
    from .session import open_session

    session = open_session()
    app = MyAssistantTUI(session, plan)
    try:
        app.run()
    finally:
        # Stop the background DB loop + dispose the engine so repeated run_tui
        # calls in one process don't leak loops / connection pools.
        session.close()
    return 0
