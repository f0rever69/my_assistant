"""Exercise real terminal selection and copy paths without touching the OS clipboard."""

import subprocess
from types import SimpleNamespace

import pytest
from my_assistant.tui import clipboard
from my_assistant.tui.app import MyAssistantTUI
from my_assistant.tui.text_export import (
    conflict_text,
    selectable_rich_text,
    transcript_text,
)
from my_assistant.tui.text_widgets import CopyableStatic, TextViewScreen
from my_assistant.tui.view_state import (
    AssistantRow,
    SystemMessage,
    SystemRow,
    ToolRow,
    UserRow,
    ViewState,
)
from rich.console import Console, Group
from rich.table import Table
from rich.text import Text
from textual.geometry import Offset
from textual.selection import Selection
from textual.widgets import Input, TextArea


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("MY_ASSISTANT_HOME", str(tmp_path / "state"))
    client = SimpleNamespace(list_models=lambda: {"models": []}, list_skills=lambda **kw: {"skills": []})
    session = SimpleNamespace(client=client, resolve_thread=lambda plan: "copy-thread", load_view_state=lambda thread: ViewState())
    result = MyAssistantTUI(session, SimpleNamespace(message=None, transparent=False))
    monkeypatch.setattr("my_assistant.tui.app.copy_native_text", lambda text: True)
    return result


def test_rich_group_tables_preserve_text_styles_and_unicode():
    table = Table.grid()
    table.add_row(Text("工具回执 中文", style="bold cyan"))
    result = selectable_rich_text(Group(Text("原文", style="red"), table), Console(), 50)
    assert isinstance(result, Text)
    assert "原文\n" in result.plain and "工具回执 中文" in result.plain
    assert result.spans


def test_transcript_export_keeps_raw_code_tool_results_and_full_token():
    token = "abcdef" * 30
    code = '```python\ndef f():\n    return "中文"\n```'
    state = ViewState(rows=(
        UserRow("修改 test.txt"), AssistantRow(code),
        ToolRow("call", "write_file", "Write", "test.txt", "沙箱已修改；本地未更新", "ok"),
        SystemRow(token),
    ))
    text = transcript_text(state)
    assert code in text and token in text and "沙箱已修改；本地未更新" in text
    assert transcript_text(state, last=True) == "[系统]\n" + token
    assert transcript_text(ViewState(), last=True) == ""


def test_conflict_text_has_multiline_diff_and_copyable_commands():
    token = "ab" * 32
    text = conflict_text({
        "path": "中文 file.txt", "host_sha256": None, "sandbox_sha256": "hash",
        "baseline_sha256": None, "resolution_token": token, "diff": "--- local\n+++ sandbox\n+中文",
    })
    assert "文件不存在" in text and "--- local\n+++ sandbox\n+中文" in text
    assert f'/workspace resolve "中文 file.txt" sandbox {token}' in text


def test_native_clipboard_uses_constant_script_and_utf8_stdin(monkeypatch):
    if clipboard.os.name != "nt":
        pytest.skip("Windows native clipboard adapter")
    calls = []
    monkeypatch.setattr(clipboard.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
    text = '中文\n`backticks` $(Get-Secret) "quotes"\n' + "abc" * 80
    assert clipboard.copy_native_text(text)
    args, kwargs = calls[0]
    assert text not in " ".join(args[0])
    assert kwargs["input"] == text.encode("utf-8")
    assert kwargs["check"] and kwargs["timeout"] == 3
    assert kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW


@pytest.mark.asyncio
async def test_main_rich_displays_are_selectable_and_ctrl_c_copies_without_interrupt(app, monkeypatch):
    interrupts = []
    monkeypatch.setattr(app, "_interrupt_run", lambda: interrupts.append(True))
    async with app.run_test(size=(100, 28)) as pilot:
        app._dispatch(SystemMessage("中文可复制\n完整令牌 " + "abcdef" * 20))
        app._flush_transcript()
        await pilot.pause()
        for selector in ("#header", "#transcript", "#status", "#progress-content"):
            widget = app.query_one(selector, CopyableStatic)
            assert widget.get_selection(Selection(None, None))[0]
        transcript = app.query_one("#transcript", CopyableStatic)
        full_text = transcript.get_selection(Selection(None, None))[0]
        app.screen.selections = {transcript: Selection(None, None)}
        app._streaming = True
        await pilot.press("ctrl+c")
        assert "中文可复制" in app.clipboard and "abcdef" * 20 in app.clipboard.replace("\n", "")
        assert app.clipboard == full_text
        assert interrupts == [] and app._streaming
        await pilot.press("escape")
        assert not app.screen.selections and interrupts == []
        await pilot.press("ctrl+c")
        assert interrupts == [True]
        app._streaming = False


@pytest.mark.asyncio
async def test_mouse_drag_actually_selects_chinese_transcript(app):
    async with app.run_test(size=(100, 28)) as pilot:
        app._dispatch(SystemMessage("可复制文字 abcdefg"))
        app._flush_transcript()
        await pilot.pause()
        widget = app.query_one("#transcript", CopyableStatic)
        start = widget.content_region.offset + Offset(2, 0)
        await pilot.mouse_down(offset=start)
        await pilot.hover(offset=start + Offset(10, 0))
        await pilot.mouse_up(offset=start + Offset(10, 0))
        await pilot.press("ctrl+c")
        assert "文字" in app.clipboard


@pytest.mark.asyncio
async def test_copy_views_keep_full_text_and_do_not_cancel_stream(app, monkeypatch):
    interrupts = []
    monkeypatch.setattr(app, "_interrupt_run", lambda: interrupts.append(True))
    raw = "```python\nprint('中文')\n```\n" + "f" * 128
    async with app.run_test(size=(80, 24)) as pilot:
        app._dispatch(SystemMessage(raw))
        await pilot.press("f8")
        assert isinstance(app.screen, TextViewScreen)
        area = app.screen.query_one(TextArea)
        assert area.read_only and raw in area.text and "线程：copy-thread" in area.text
        await pilot.press("ctrl+a", "ctrl+c")
        assert app.clipboard == area.text
        await pilot.press("escape")
        assert len(app.screen_stack) == 1
        app._streaming = True
        app._handle_submit("/copy last")
        await pilot.pause()
        assert app.screen.query_one(TextArea).text == "[系统]\n" + raw
        # Background updates must not change this reviewed/copied snapshot.
        app._dispatch(SystemMessage("new stream output"))
        app._flush_transcript()
        app._refresh_status()
        await pilot.pause()
        await pilot.click("#copy-all")
        assert app.clipboard == "[系统]\n" + raw
        assert app._streaming and interrupts == []
        await pilot.press("escape")
        app._streaming = False
        app._handle_submit("/copy invalid")
        assert len(app.screen_stack) == 1
        assert "Usage: /copy" in app.state.rows[-1].text


@pytest.mark.asyncio
async def test_composer_selection_can_copy_without_quitting(app):
    async with app.run_test() as pilot:
        composer = app.query_one("#composer", Input)
        composer.value = "copy composer 中文"
        composer.select_all()
        await pilot.press("ctrl+c")
        assert app.clipboard == composer.value


@pytest.mark.asyncio
async def test_native_clipboard_failure_keeps_textual_fallback(app, monkeypatch):
    def fail(text):
        raise subprocess.TimeoutExpired("clipboard", 3)

    monkeypatch.setattr("my_assistant.tui.app.copy_native_text", fail)
    async with app.run_test():
        app.copy_to_clipboard("中文 fallback")
        assert app.clipboard == "中文 fallback"
