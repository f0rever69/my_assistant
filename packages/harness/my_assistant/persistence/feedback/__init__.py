"""Feedback persistence — ORM and SQL repository."""

from my_assistant.persistence.feedback.model import FeedbackRow
from my_assistant.persistence.feedback.sql import FeedbackRepository

__all__ = ["FeedbackRepository", "FeedbackRow"]
