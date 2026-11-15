from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RagConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_mode: Literal["strict", "annotate"] = "strict"
    model: str | None = None
    verifier_model: str | None = None
    allow_model_data: bool = False
    audit_max_runs: int = Field(default=500, ge=1, le=10_000)
