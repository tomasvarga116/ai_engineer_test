"""Domain models for extracted clauses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Clause(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, frozen=True)

    id: str = Field(..., description="Clause number as printed, e.g. '1', '17'.")
    title: str = Field(..., description="Clause heading.")
    text: str = Field(..., description="Clause body, with strikethrough text removed.")

    @field_validator("id")
    @classmethod
    def _id_nonempty(cls, v: str) -> str:
        if not v:
            raise ValueError("id must be non-empty")
        return v

    @field_validator("title", "text")
    @classmethod
    def _text_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be non-empty")
        return v


class ClauseDocument(BaseModel):
    source_pdf: str
    page_range: tuple[int, int]
    clauses: list[Clause]

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)
