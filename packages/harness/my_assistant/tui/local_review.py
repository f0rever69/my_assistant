"""Explicit desktop file review actions for the terminal UI."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

_DOCUMENT_SUFFIXES = frozenset({
    ".txt", ".md", ".rst", ".log", ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml",
    ".docx", ".pdf", ".odt", ".xlsx", ".ods", ".pptx", ".odp",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff",
})


def _launch(args: list[str]) -> None:
    subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def open_document(path: Path) -> None:
    if path.suffix.lower() not in _DOCUMENT_SUFFIXES:
        raise ValueError("此类型请复制路径后在编辑器中手动打开。")
    if sys.platform == "win32":
        os.startfile(str(path), "open")
        return
    opener = shutil.which("open" if sys.platform == "darwin" else "xdg-open")
    if opener is None:
        raise FileNotFoundError("未找到桌面打开程序，请复制文件路径。")
    _launch([opener, str(path)])


def open_editor_diff(original: Path, candidate: Path) -> None:
    if sys.platform == "win32":
        editor = shutil.which("Code.exe")
        launcher = shutil.which("code.cmd")
        if editor is None and launcher:
            executable = Path(launcher).parent.parent / "Code.exe"
            if executable.is_file():
                editor = str(executable)
    else:
        editor = shutil.which("code")
    if editor is None:
        raise FileNotFoundError("未找到 VS Code。请复制两份文件的路径，在编辑器中对照。")
    _launch([editor, "--diff", str(original), str(candidate)])
