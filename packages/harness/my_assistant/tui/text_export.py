"""Pure exports for selectable terminal text and conflict-review snapshots."""

from rich.console import Console
from rich.text import Text

from .view_state import ToolRow, ViewState


def selectable_rich_text(renderable, console: Console, width: int) -> Text:
    """Flatten Rich groups/tables to styled Text, which Textual can select.

    Keep rendered colors and code formatting while discarding terminal controls.
    Textual's default selection supports Text/Content, not arbitrary Rich groups.
    """
    text = Text()
    for segment in console.render(renderable, console.options.update(width=max(1, width))):
        if not segment.control:
            text.append(segment.text, style=segment.style)
    return text


def transcript_text(state: ViewState, *, last: bool = False) -> str:
    rows = state.rows[-1:] if last else state.rows
    labels = {"user": "用户", "assistant": "助手", "system": "系统", "tool": "工具"}
    parts = []
    for row in rows:
        if isinstance(row, ToolRow):
            parts.append(f"[工具 {row.title} / {row.status}]\n{row.detail}\n{row.result}".rstrip())
        else:
            parts.append(f"[{labels[row.kind]}]\n{row.text}")
    return "\n\n".join(parts)


def conflict_text(review: dict) -> str:
    name = review["path"]
    quoted = '"' + name.replace('"', '\\"') + '"'
    token = review["resolution_token"]
    return (
        f"文件：{name}\n"
        f"本地 SHA-256：{review['host_sha256'] or '文件不存在'}\n"
        f"沙箱 SHA-256：{review['sandbox_sha256'] or '文件不存在'}\n"
        f"同步基线：{review['baseline_sha256'] or '基线无此文件'}\n\n"
        f"{review['diff']}\n\n"
        "保留本地版本：\n"
        f"/workspace resolve {quoted} local {token}\n\n"
        "采用沙箱版本并写回本地：\n"
        f"/workspace resolve {quoted} sandbox {token}\n"
    )
