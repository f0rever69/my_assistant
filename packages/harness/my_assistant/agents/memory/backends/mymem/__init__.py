"""MyMem backend -- the default local memory manager.

The adapter wraps its manager class (:mod:`my_mem`) around the self-contained
MyMem engine in ``mymem/``. Shared summaries and agent facts retain their
existing on-disk format.
"""

from .my_mem import MyMem

#: The :class:`~my_assistant.agents.memory.manager.MemoryManager` subclass this
#: backend exposes. Discovered by the factory's ``_scan_backends`` drop-in
#: mechanism under the folder name ``mymem``.
MANAGER_CLASS = MyMem
