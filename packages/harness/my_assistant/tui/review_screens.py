"""Dialogs for inspecting exact file versions and confirming a Workspace review."""

from collections import Counter
from collections.abc import Callable
from typing import ClassVar

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, SelectionList, TextArea
from textual.widgets.selection_list import Selection

from my_assistant.sandbox.workspace_review import FileComparison, WorkspaceReviewChange

from .local_review import open_document, open_editor_diff
from .text_widgets import CopyableStatic, TextViewScreen

_KIND_LABELS = {"added": "新增", "modified": "修改", "deleted": "删除"}


class WorkspaceReviewScreen(ModalScreen):
    DEFAULT_CSS = """
    WorkspaceReviewScreen { align: center middle; }
    WorkspaceReviewScreen #review-dialog {
        width: 96%; height: 92%; background: $surface; border: round $accent; padding: 1;
    }
    WorkspaceReviewScreen Static { height: auto; }
    WorkspaceReviewScreen #review-files { height: 1fr; min-height: 4; margin: 1 0; }
    WorkspaceReviewScreen #review-actions { height: auto; margin-top: 1; }
    WorkspaceReviewScreen Button { width: 1fr; min-width: 0; margin: 0 1; }
    """
    BINDINGS: ClassVar = [Binding("escape", "cancel", "取消")]

    def __init__(
        self, changes: list[WorkspaceReviewChange], checks: list[dict],
        compare: Callable[[WorkspaceReviewChange], None],
    ):
        super().__init__()
        self.changes = {item.path: item for item in changes}
        self.checks = checks
        self.compare = compare

    def compose(self) -> ComposeResult:
        counts = Counter(item.kind for item in self.changes.values())
        summary = " · ".join(f"{label} {counts[kind]}" for kind, label in _KIND_LABELS.items())
        blocked = sum(bool(item.blocked_reason) for item in self.changes.values())
        if blocked:
            summary += f" · 不可勾选 {blocked}"
        checks = Counter(check.get("status") for check in self.checks)
        evidence = (
            f"已记录的任务验收：通过 {checks['passed']} · 失败 {checks['failed']} · 其余 {len(self.checks) - checks['passed'] - checks['failed']}"
            if checks else "当前没有任务验收记录"
        )
        rows = [
            Selection(
                escape(f"[{_KIND_LABELS[item.kind]}] {item.path}" + (f" · {item.blocked_reason}" if item.blocked_reason else "")),
                item.path, disabled=bool(item.blocked_reason),
            )
            for item in self.changes.values()
        ]
        with Vertical(id="review-dialog"):
            yield CopyableStatic("审阅原工作区变更\n" + summary, markup=False)
            yield CopyableStatic(evidence, markup=False)
            yield SelectionList(*rows, id="review-files")
            yield CopyableStatic("勾选要写回的文件。删除项写回时会备份原文件。", id="review-selection", markup=False)
            with Horizontal(id="review-actions"):
                yield Button("查看对照", id="review-compare")
                yield Button("查看验收", id="review-checks", disabled=not self.checks)
                yield Button("写回选中", id="review-apply", variant="warning", disabled=True)
                yield Button("取消", id="review-cancel")

    def on_mount(self):
        self.query_one(SelectionList).focus()

    def on_selection_list_selected_changed(self, event: SelectionList.SelectedChanged):
        selected = event.selection_list.selected
        deletions = sum(self.changes[path].kind == "deleted" for path in selected)
        summary = f"已勾选 {len(selected)} 项，其中删除 {deletions} 项。确认后修改原工作区。"
        self.query_one("#review-selection", CopyableStatic).update(summary)
        self.query_one("#review-apply", Button).disabled = not selected

    def action_cancel(self):
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed):
        event.stop()
        choices = self.query_one(SelectionList)
        match event.button.id:
            case "review-compare":
                if choices.highlighted_option is not None:
                    self.compare(self.changes[choices.highlighted_option.value])
            case "review-checks":
                text = "\n".join(
                    f"{item.get('criterion_id', '')} · {item.get('status', '')}\n{item.get('detail', '')}"
                    for item in self.checks
                )
                self.app.push_screen(TextViewScreen(text, "已记录的任务验收结果"))
            case "review-apply":
                selected = {path: self.changes[path].token for path in choices.selected if not self.changes[path].blocked_reason}
                if selected:
                    self.dismiss(selected)
            case "review-cancel":
                self.action_cancel()


class FileComparisonScreen(TextViewScreen):
    """Inspect a stable pair of copies without opening an editable original."""

    def __init__(self, comparison: FileComparison):
        self.comparison = comparison
        sides = []
        for label, path, size in (
            ("原稿", comparison.original, comparison.original_size),
            ("候选稿", comparison.candidate, comparison.candidate_size),
        ):
            sides.append(f"{label}（{size:,} 字节）\n{path}" if path else f"{label}不存在")
        text = "\n\n".join([*sides, comparison.text_preview])
        super().__init__(text, f"文件对照 · {comparison.path}")

    def compose(self) -> ComposeResult:
        comparison = self.comparison
        with Vertical(classes="text-dialog"):
            yield CopyableStatic(self.title_text, classes="text-title", markup=False)
            yield CopyableStatic(
                "打开完整审阅副本。DOCX 可在 Word 的「审阅 → 比较」中选择两份副本。",
                classes="text-help", markup=False,
            )
            yield TextArea(self.text, read_only=True, soft_wrap=True, id="copy-text")
            with Horizontal(classes="text-actions"):
                yield Button("打开原稿", id="open-original", disabled=comparison.original is None)
                yield Button("打开候选稿", id="open-candidate", disabled=comparison.candidate is None)
                if comparison.is_text and comparison.original and comparison.candidate:
                    yield Button("编辑器对照", id="open-diff")
                yield Button("复制详情", id="copy-all")
                yield Button("关闭", id="text-close")

    def _handle_button(self, button_id):
        comparison = self.comparison
        path = {"open-original": comparison.original, "open-candidate": comparison.candidate}.get(button_id)
        try:
            if button_id == "open-diff":
                if comparison.original and comparison.candidate:
                    open_editor_diff(comparison.original, comparison.candidate)
            elif button_id in {"open-original", "open-candidate"}:
                if path is not None:
                    open_document(path)
            else:
                super()._handle_button(button_id)
        except (OSError, ValueError) as exc:
            self.notify(str(exc), severity="error", timeout=5)
