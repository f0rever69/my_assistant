"""Resolve user-visible output files within one thread's host mount."""

import stat
from dataclasses import replace
from pathlib import Path

from my_assistant.config.paths import VIRTUAL_PATH_PREFIX, get_paths
from my_assistant.config.workspace import Workspace
from my_assistant.constants import TOOL_RESULTS_DIRNAME
from my_assistant.sandbox.workspace_copy import _walk

OUTPUTS_PREFIX = f"{VIRTUAL_PATH_PREFIX}/outputs/"


def output_scope(workspace: Workspace | None, thread_id: str, user_id: str) -> Workspace:
    if workspace is not None:
        path = workspace.resolve(workspace.runtime_dir(thread_id) / "outputs")
    else:
        paths = get_paths()
        state = Workspace.from_path(paths.base_dir)
        path = state.resolve(paths.sandbox_outputs_dir(thread_id, user_id=user_id))
    return Workspace.from_path(path)


def resolve_output_file(scope: Workspace, name: str, storage_subdir: str = TOOL_RESULTS_DIRNAME) -> Path:
    relative = Path(name.removeprefix(OUTPUTS_PREFIX))
    private = {TOOL_RESULTS_DIRNAME.casefold(), storage_subdir.casefold()}
    if (
        not relative.parts or relative.anchor or ".." in relative.parts
        or "\\" in name or ":" in name or relative.parts[0].casefold() in private
    ):
        raise PermissionError("请选择当前线程输出目录中的文件。")
    path = scope.check(relative)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise PermissionError("只能打开输出目录中的普通文件。")
    return path


def list_output_files(scope: Workspace, storage_subdir: str) -> list[dict]:
    outputs = []
    root = scope.root
    internal = tuple(dict.fromkeys((TOOL_RESULTS_DIRNAME, storage_subdir)))
    visible_scope = replace(scope, ignored_patterns=(*scope.ignored_patterns, *internal))
    for file in _walk(root, visible_scope):
        relative = file.relative_to(root).as_posix()
        try:
            path = resolve_output_file(scope, relative, storage_subdir)
            size = path.stat().st_size
        except (OSError, ValueError):
            continue
        outputs.append({"path": relative, "size": size})
    return outputs
