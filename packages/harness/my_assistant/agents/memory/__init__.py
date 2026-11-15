"""Pluggable memory for MyAssistant.

The shared, backend-agnostic core: the :class:`MemoryManager` contract, the
:func:`get_memory_manager` singleton factory, and :func:`reset_memory_manager`.
Backends live under :mod:`backends` (each exposing ``MANAGER_CLASS``). The
default MyMem adapter lives in ``backends/mymem/`` and wraps the vendored
MyMem engine. Swap backend = drop a ``backends/<name>/`` folder + set
``MemoryConfig.manager_class`` -- nothing else in MyAssistant changes.

MyMem-private symbols (``format_memory_for_injection``, ``get_memory_data``,
``MemoryUpdater``, ``FileMemoryStorage``, ...) are NOT re-exported here -- import
them directly from ``my_assistant.agents.memory.backends.mymem.mymem.core.*``.
"""

from my_assistant.agents.memory.manager import (
    MemoryConflictError,
    MemoryCorruptionError,
    MemoryManager,
    MemoryManagerError,
    MemoryReadError,
    get_memory_manager,
    memory_read_failures_are_fatal,
    reset_memory_manager,
)

__all__ = [
    "MemoryManager",
    "MemoryManagerError",
    "MemoryReadError",
    "MemoryConflictError",
    "MemoryCorruptionError",
    "get_memory_manager",
    "memory_read_failures_are_fatal",
    "reset_memory_manager",
]
