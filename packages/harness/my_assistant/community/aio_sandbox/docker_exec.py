"""Bound local Docker commands by terminating their container on a timeout."""

import json
import math
import subprocess
import tempfile
import threading

_MAX_CAPTURE = 1024 * 1024
_RUN_COMMAND = (
    "import json, os, sys\n"
    "command, env = json.load(sys.stdin)\n"
    "os.execve('/bin/bash', ['bash', '-c', command], os.environ | env)\n"
)


def execute_isolated_command(
    container_ref: str,
    default_timeout: float,
    command: str,
    env: dict[str, str] | None,
    timeout: float | None,
) -> str:
    deadline = float(default_timeout if timeout is None else timeout)
    if not math.isfinite(deadline) or deadline <= 0:
        raise ValueError("Docker command timeout must be positive and finite")
    checked = subprocess.run(
        ["docker", "container", "inspect", "--format", "{{.Id}}", container_ref],
        capture_output=True,
        timeout=10,
        check=False,
    )
    if checked.returncode != 0:
        return "Error: Docker sandbox container is unavailable"
    container_id = checked.stdout.decode().strip()
    command_args = [
        "docker",
        "exec",
        "-i",
        "--user",
        "gem",
        "--env",
        "HOME=/home/gem",
        "--workdir",
        "/mnt/user-data/workspace",
        container_id,
        "python3",
        "-I",
        "-c",
        _RUN_COMMAND,
    ]
    captured = bytearray()
    truncated = False

    def drain(pipe):
        nonlocal truncated
        for chunk in iter(lambda: pipe.read(65536), b""):
            remaining = max(0, _MAX_CAPTURE - len(captured))
            captured.extend(chunk[:remaining])
            truncated |= len(chunk) > remaining

    with tempfile.TemporaryFile() as stdin:
        stdin.write(json.dumps([command, env or {}]).encode())
        stdin.seek(0)
        with subprocess.Popen(
            command_args, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        ) as process:
            reader = threading.Thread(target=drain, args=(process.stdout,), daemon=True)
            reader.start()
            try:
                exit_code = process.wait(timeout=deadline)
            except subprocess.TimeoutExpired:
                process.kill()
                stopped = subprocess.run(
                    ["docker", "kill", container_id],
                    capture_output=True,
                    timeout=15,
                    check=False,
                )
                if stopped.returncode == 0:
                    return f"Error: command timed out after {deadline:g}s; Docker sandbox terminated"
                raise RuntimeError(
                    "Docker command timed out and container termination could not be confirmed"
                )
            finally:
                reader.join(timeout=2)
    output = captured.decode("utf-8", errors="replace").strip()
    if truncated:
        output += "\n[output truncated at 1 MiB]"
    if exit_code != 0:
        return f"{output}\nExit Code: {exit_code}".strip()
    return output or "(no output)"
