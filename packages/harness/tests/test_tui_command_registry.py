from my_assistant.tui.command_registry import (
    BUILTIN_COMMANDS,
    format_command_help,
    resolve,
)


def test_saved_threads_have_one_resume_command():
    builtin_names = {command.name for command in BUILTIN_COMMANDS}

    assert "resume" in builtin_names
    assert "threads" not in builtin_names
    assert "switch" not in builtin_names
    assert "/resume" in format_command_help()
    assert "/threads" not in format_command_help()


def test_resume_supports_picker_and_direct_reference():
    picker = resolve("/resume")
    direct = resolve("/resume project notes")

    assert (picker.kind, picker.name, picker.args) == ("builtin", "resume", "")
    assert (direct.kind, direct.name, direct.args) == (
        "builtin",
        "resume",
        "project notes",
    )
    assert resolve("/threads").kind == "unknown"
    assert resolve("/switch").kind == "unknown"
