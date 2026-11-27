"""Native Windows clipboard with input kept out of shell command text."""

import os
import subprocess
from pathlib import Path


def copy_native_text(text: str) -> bool:
    """Return True when a native copy succeeded; other platforms use OSC 52.

    PowerShell's clipboard cmdlet handles Unicode and clipboard ownership.
    The script is constant: untrusted text, quotes and command substitutions
    travel exclusively on UTF-8 stdin, never as executable command text.
    """
    if os.name != "nt":
        return False
    executable = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    script = (
        "$ErrorActionPreference = 'Stop'; "
        "[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false); "
        "Set-Clipboard -Value ([Console]::In.ReadToEnd())"
    )
    subprocess.run(
        [str(executable), "-NoProfile", "-NonInteractive", "-Command", script],
        input=text.encode("utf-8"), capture_output=True,
        check=True, timeout=3, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return True
