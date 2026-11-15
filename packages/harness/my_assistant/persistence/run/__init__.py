"""Run metadata persistence — ORM and SQL repository."""

from my_assistant.persistence.run.model import RunRow
from my_assistant.persistence.run.sql import RunRepository

__all__ = ["RunRepository", "RunRow"]
