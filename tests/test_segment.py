"""Tests for LLMSegmenter, with the OpenAI client mocked."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from charter_parser.pdf_extract import Line, PageContent
from charter_parser.segment import LLMSegmenter, _SECTIONS, diagnostic_dump


def _page(lines: list[str], page_no: int) -> PageContent:
    return PageContent(
        page_number=page_no,
        lines=[Line(text=t, size=10.0, page=page_no, top=float(i)) for i, t in enumerate(lines)],
    )


def _stream_chunks_for(args_payload: dict) -> list[SimpleNamespace]:
    """Fake streamed response that delivers a tool call in two argument chunks."""
    raw = json.dumps(args_payload)
    half = len(raw) // 2
    chunks: list[SimpleNamespace] = []

    def make_delta_chunk(arg_fragment: str) -> SimpleNamespace:
        function = SimpleNamespace(name="emit_clauses", arguments=arg_fragment)
        tool_call = SimpleNamespace(index=0, id="call_test", type="function", function=function)
        delta = SimpleNamespace(tool_calls=[tool_call])
        choice = SimpleNamespace(delta=delta, finish_reason=None, index=0)
        return SimpleNamespace(choices=[choice], usage=None)

    chunks.append(make_delta_chunk(raw[:half]))
    chunks.append(make_delta_chunk(raw[half:]))
    final_choice = SimpleNamespace(
        delta=SimpleNamespace(tool_calls=None), finish_reason="tool_calls", index=0
    )
    final_usage = SimpleNamespace(prompt_tokens=120, completion_tokens=60, total_tokens=180)
    chunks.append(SimpleNamespace(choices=[final_choice], usage=final_usage))
    return chunks


def _client_returning(per_call_payloads: list[dict]) -> MagicMock:
    client = MagicMock()
    streams = [iter(_stream_chunks_for(p)) for p in per_call_payloads]
    client.chat.completions.create.side_effect = streams
    return client


def _all_section_pages() -> list[PageContent]:
    return [_page(["dummy"], page_no=s.first_page) for s in _SECTIONS]


def test_segment_calls_one_request_per_section_in_order() -> None:
    payloads = [
        {"clauses": [{"id": "1", "title": f"sec{i}-1", "text": "body"}]} for i in range(len(_SECTIONS))
    ]
    client = _client_returning(payloads)
    segmenter = LLMSegmenter(client=client)
    clauses = segmenter.segment(_all_section_pages())

    assert client.chat.completions.create.call_count == len(_SECTIONS)
    assert [c.title for c in clauses] == [f"sec{i}-1" for i in range(len(_SECTIONS))]
    assert segmenter.stats.sections_called == len(_SECTIONS)


def test_segment_skips_sections_with_no_pages() -> None:
    pages = [_page(["dummy"], page_no=_SECTIONS[1].first_page)]
    client = _client_returning(
        [{"clauses": [{"id": "1", "title": "only", "text": "body"}]}]
    )
    segmenter = LLMSegmenter(client=client)
    clauses = segmenter.segment(pages)
    assert client.chat.completions.create.call_count == 1
    assert [c.title for c in clauses] == ["only"]


def test_segment_returns_empty_for_empty_input() -> None:
    client = MagicMock()
    segmenter = LLMSegmenter(client=client)
    assert segmenter.segment([]) == []
    client.chat.completions.create.assert_not_called()


def test_segment_skips_malformed_clauses_but_keeps_valid_ones() -> None:
    payloads = [
        {
            "clauses": [
                {"id": "1", "title": "Good", "text": "Body."},
                {"id": "", "title": "Empty id", "text": "Body."},
                {"id": "2", "title": "Also good", "text": "Body."},
            ]
        }
    ]
    client = _client_returning(payloads)
    segmenter = LLMSegmenter(client=client)
    pages = [_page(["dummy"], page_no=_SECTIONS[0].first_page)]
    clauses = segmenter.segment(pages)
    assert [c.id for c in clauses] == ["1", "2"]


def test_segment_raises_when_no_tool_call_emitted() -> None:
    client = MagicMock()
    final_choice = SimpleNamespace(
        delta=SimpleNamespace(tool_calls=None), finish_reason="stop", index=0
    )
    final_chunk = SimpleNamespace(
        choices=[final_choice],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=0, total_tokens=10),
    )
    client.chat.completions.create.return_value = iter([final_chunk])
    segmenter = LLMSegmenter(client=client)
    pages = [_page(["dummy"], page_no=_SECTIONS[0].first_page)]
    with pytest.raises(RuntimeError, match="no tool call"):
        segmenter.segment(pages)


def test_segment_raises_on_invalid_json_arguments() -> None:
    client = MagicMock()
    function = SimpleNamespace(name="emit_clauses", arguments="not json {")
    tool_call = SimpleNamespace(index=0, id="call_test", type="function", function=function)
    delta_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(tool_calls=[tool_call]), finish_reason=None, index=0
            )
        ],
        usage=None,
    )
    final_chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(tool_calls=None), finish_reason="tool_calls", index=0
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10, total_tokens=20),
    )
    client.chat.completions.create.return_value = iter([delta_chunk, final_chunk])
    segmenter = LLMSegmenter(client=client)
    pages = [_page(["dummy"], page_no=_SECTIONS[0].first_page)]
    with pytest.raises(RuntimeError, match="not valid JSON"):
        segmenter.segment(pages)


def test_diagnostic_dump_includes_page_markers() -> None:
    pages = [_page(["1. PREAMBLE", "Body."], page_no=6), _page(["2. CONDITIONS"], page_no=7)]
    dumped = diagnostic_dump(pages)
    assert "[PAGE 6]" in dumped
    assert "[PAGE 7]" in dumped
    assert "1. PREAMBLE" in dumped
    assert "2. CONDITIONS" in dumped
