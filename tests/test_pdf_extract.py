"""Tests for the pure helpers in pdf_extract."""

from __future__ import annotations

from charter_parser.pdf_extract import _GUTTER_NUMBER_RE, _assemble_line_text


def test_gutter_regex_matches_bare_line_numbers() -> None:
    assert _GUTTER_NUMBER_RE.match("1")
    assert _GUTTER_NUMBER_RE.match("  42  ")
    assert _GUTTER_NUMBER_RE.match("1, 2, 3")


def test_gutter_regex_rejects_real_text() -> None:
    assert _GUTTER_NUMBER_RE.match("Clause 1") is None
    assert _GUTTER_NUMBER_RE.match("1. PREAMBLE") is None
    assert _GUTTER_NUMBER_RE.match("Page 1 of 12") is None


def _char(text: str, x0: float, x1: float) -> dict:
    return {"text": text, "x0": x0, "x1": x1}


def test_assemble_line_inserts_space_for_word_gap() -> None:
    chars = [
        _char("H", 0.0, 5.0),
        _char("i", 5.0, 7.0),
        _char("t", 30.0, 33.0),
        _char("o", 33.0, 38.0),
    ]
    assert _assemble_line_text(chars) == "Hi to"


def test_assemble_line_no_space_for_tight_glyphs() -> None:
    chars = [
        _char("a", 0.0, 5.0),
        _char("b", 5.1, 10.0),
        _char("c", 10.1, 15.0),
    ]
    assert _assemble_line_text(chars) == "abc"
