"""Tests for the pydantic domain models."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from charter_parser.models import Clause, ClauseDocument


def test_clause_strips_whitespace() -> None:
    c = Clause(id="  1 ", title="  PREAMBLE  ", text="  Body text. ")
    assert c.id == "1"
    assert c.title == "PREAMBLE"
    assert c.text == "Body text."


def test_clause_rejects_empty_fields() -> None:
    with pytest.raises(ValidationError):
        Clause(id="", title="x", text="y")
    with pytest.raises(ValidationError):
        Clause(id="1", title="   ", text="y")
    with pytest.raises(ValidationError):
        Clause(id="1", title="x", text="")


def test_document_serialises_to_expected_json_shape() -> None:
    doc = ClauseDocument(
        source_pdf="example.pdf",
        page_range=(6, 39),
        clauses=[
            Clause(id="1", title="PREAMBLE", text="Body."),
            Clause(id="2", title="CONDITIONS", text="Other body."),
        ],
    )
    payload = json.loads(doc.to_json())
    assert payload["source_pdf"] == "example.pdf"
    assert payload["page_range"] == [6, 39]
    assert [c["id"] for c in payload["clauses"]] == ["1", "2"]
    assert set(payload["clauses"][0]) == {"id", "title", "text"}
