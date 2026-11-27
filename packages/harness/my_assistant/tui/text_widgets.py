"""Selectable Rich displays and copyable, keyboard-accessible review dialogs."""

from typing import ClassVar

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea

from .text_export import conflict_text, selectable_rich_text


class CopyableStatic(Static):
    """Preserve Rich layouts while exposing actual text to native selection."""

    ALLOW_SELECT = True

    def __init__(self, content="", **kwargs):
        self._copy_source = content
        self._copy_width = 0
        super().__init__(content, **kwargs)

    def update(self, content="", *, layout=True):
        self._copy_source = content
        self._update_text(layout=layout)

    def _update_text(self, *, layout=True):
        width = self.content_size.width or self.app.size.width or 80
        self._copy_width = width
        source = self._copy_source
        if not isinstance(source, (str, Text)):
            source = selectable_rich_text(source, self.app.console, width)
        super().update(source, layout=layout)

    def on_mount(self):
        self._update_text()

    def on_resize(self):
        if self.content_size.width and self.content_size.width != self._copy_width:
            self._update_text()

    def get_selection(self, selection):
        try:
            return super().get_selection(selection)
        except IndexError:
            # A streaming update may shorten the text while it is selected.
            return None


class TextViewScreen(ModalScreen):
    DEFAULT_CSS = """
    TextViewScreen { align: center middle; }
    TextViewScreen .text-dialog {
        width: 94%; height: 90%; border: round $accent;
        background: $surface; padding: 1;
    }
    TextViewScreen .text-title { height: auto; text-style: bold; margin-bottom: 1; }
    TextViewScreen .text-help { height: auto; margin-bottom: 1; }
    TextViewScreen TextArea { height: 1fr; min-height: 4; }
    TextViewScreen .text-actions { height: auto; margin-top: 1; }
    TextViewScreen Button { width: 1fr; min-width: 0; margin: 0 1; }
    """
    BINDINGS: ClassVar = [
        Binding("escape", "cancel", "关闭"),
        Binding("ctrl+a", "select_all", "全选", priority=True),
        Binding("ctrl+shift+c", "copy_all", "复制全文"),
    ]

    def __init__(self, text: str, title: str = "可复制文本"):
        super().__init__()
        self.text = text
        self.title_text = title

    def compose(self) -> ComposeResult:
        with Vertical(classes="text-dialog"):
            yield CopyableStatic(self.title_text, classes="text-title", markup=False)
            yield CopyableStatic("鼠标选择或 Ctrl+A 全选，Ctrl+C 复制；Esc 关闭。", classes="text-help", markup=False)
            yield TextArea(self.text, read_only=True, soft_wrap=True, id="copy-text")
            with Horizontal(classes="text-actions"):
                yield Button("复制全文", id="copy-all", variant="primary")
                yield Button("关闭", id="text-close")

    def on_mount(self):
        self._focus_initial()

    def _focus_initial(self):
        self.query_one(TextArea).focus()

    def action_select_all(self):
        area = self.query_one(TextArea)
        area.focus()
        area.select_all()

    def action_copy_all(self):
        self.app.copy_to_clipboard(self.text)

    def action_cancel(self):
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed):
        event.stop()
        self._handle_button(event.button.id)

    def _handle_button(self, button_id):
        if button_id == "copy-all":
            self.action_copy_all()
        else:
            self.action_cancel()


class ConflictReviewScreen(TextViewScreen):
    """Return a user choice; the owner holds the reviewed token and scope."""

    DEFAULT_CSS = """
    ConflictReviewScreen { align: center middle; }
    ConflictReviewScreen .text-dialog {
        width: 94%; height: 90%; border: round $accent;
        background: $surface; padding: 1;
    }
    ConflictReviewScreen .text-title { height: auto; text-style: bold; margin-bottom: 1; }
    ConflictReviewScreen .text-help { height: auto; margin-bottom: 1; }
    ConflictReviewScreen TextArea { height: 1fr; min-height: 4; }
    ConflictReviewScreen .text-actions { height: auto; margin-top: 1; }
    ConflictReviewScreen Button { width: 1fr; min-width: 0; margin: 0 1; }
    """
    BINDINGS: ClassVar = [
        Binding("l", "keep_local", "保留本地", priority=True),
        Binding("s", "keep_sandbox", "采用沙箱写回", priority=True),
    ]

    def __init__(self, review: dict):
        super().__init__(conflict_text(review), "审阅文件版本并选择保留方式")

    def compose(self) -> ComposeResult:
        with Vertical(classes="text-dialog"):
            yield CopyableStatic(self.title_text, classes="text-title", markup=False)
            yield CopyableStatic(
                "L 保留本地并覆盖沙箱；S 采用沙箱并写回本地。选择不存在的一侧表示删除。\n"
                "程序自动携带审核令牌；覆盖前备份双方版本。Esc 取消，Ctrl+A/C 复制。",
                classes="text-help", markup=False,
            )
            yield TextArea(self.text, read_only=True, soft_wrap=True, id="copy-text")
            with Horizontal(classes="text-actions"):
                yield Button("保留本地", id="conflict-local")
                yield Button("采用沙箱写回", id="conflict-sandbox", variant="warning")
                yield Button("复制详情", id="copy-all")
                yield Button("取消", id="conflict-cancel")

    def _focus_initial(self):
        # Opening a review and pressing Enter must not overwrite anything.
        self.query_one("#conflict-cancel", Button).focus()

    def action_keep_local(self):
        self.dismiss("local")

    def action_keep_sandbox(self):
        self.dismiss("sandbox")

    def _handle_button(self, button_id):
        if button_id == "conflict-local":
            self.action_keep_local()
        elif button_id == "conflict-sandbox":
            self.action_keep_sandbox()
        elif button_id == "copy-all":
            self.action_copy_all()
        else:
            self.action_cancel()
