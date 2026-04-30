"""LLM-driven clause segmentation."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import openai

from charter_parser.models import Clause
from charter_parser.pdf_extract import PageContent

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "google/gemini-2.5-flash")
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")


@dataclass(frozen=True)
class _Section:
    label: str
    first_page: int
    last_page: int
    style_hint: str


# Each section restarts numbering at 1 and uses a different visual convention,
# so we extract them independently and concatenate.
_SECTIONS: tuple[_Section, ...] = (
    _Section(
        label="Base form (SHELLVOY 5 / ASBATANKVOY)",
        first_page=6,
        last_page=17,
        style_hint=(
            "Two-column layout. Clause titles appear in the LEFT MARGIN GUTTER as short "
            "fragments split across one or two lines (e.g. 'Condition / Of vessel', "
            "'Cleanliness / Of tanks'). The body begins on a line starting with the "
            "clause number to the right (e.g. '1. Owners shall exercise...'). "
            "Reassemble the gutter fragment into one human-readable title."
        ),
    ),
    _Section(
        label="Shell Additional Clauses",
        first_page=18,
        last_page=35,
        style_hint=(
            "Title-case headings on their own line, e.g. '1. Indemnity Clause'. "
            "Numbering restarts at 1. Internal sub-items like '1) Oil Pollution' "
            "belong to the parent clause and are NOT separate clauses."
        ),
    ),
    _Section(
        label="Charterers' Additional Clauses",
        first_page=36,
        last_page=39,
        style_hint=(
            "ALL-CAPS headings on their own line, e.g. '1. INTERNATIONAL REGULATIONS "
            "CLAUSE'. Numbering restarts at 1."
        ),
    ),
)


_SYSTEM_PROMPT = """\
You extract numbered legal clauses from one section of a maritime charter party.

Input notes:
- Strikethrough text has already been removed at the character layer; treat
  anything you see as intended.
- Bare line numbers from the PDF gutter have already been stripped.
- You will receive a section style hint and the cleaned page text wrapped
  in PAGE markers.

Output rules:
- One entry per top-level numbered clause, in source order.
- id   = the clause number as printed, e.g. "1", "17".
- title = the heading only (no body sentences).
- text  = the full body, verbatim. Mend mid-word line breaks. Preserve
  paragraph breaks as blank lines. Do not paraphrase, summarise, or reorder.
- Sub-items like "(a)", "(i)", "1)" belong to the enclosing clause.
- Do not emit section banners (e.g. "SHELL ADDITIONAL CLAUSES") as clauses.
- If a clause body is genuinely empty (e.g. "Not Applicable"), use that
  marker phrase as the text.

Call emit_clauses exactly once with the full ordered list. Do not respond
in plain prose.
"""


_EMIT_CLAUSES_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "emit_clauses",
        "description": "Emit the ordered list of clauses extracted from this section.",
        "parameters": {
            "type": "object",
            "properties": {
                "clauses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "text": {"type": "string"},
                        },
                        "required": ["id", "title", "text"],
                    },
                }
            },
            "required": ["clauses"],
        },
    },
}


@dataclass
class SegmentationStats:
    input_tokens: int = 0
    output_tokens: int = 0
    sections_called: int = 0
    per_section: list[tuple[str, int, int, int]] = field(default_factory=list)


class LLMSegmenter:
    """Segment cleaned page text into ``Clause`` objects via an LLM."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        client: openai.OpenAI | None = None,
        max_tokens: int = 16384,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._client = client
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.stats = SegmentationStats()

    @property
    def client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(
                base_url=self.base_url,
                default_headers={
                    "HTTP-Referer": "https://github.com/local/charter-parser",
                    "X-Title": "Charter Party Clause Parser",
                },
                timeout=self.timeout,
            )
        return self._client

    def segment(self, pages: Sequence[PageContent]) -> list[Clause]:
        if not pages:
            return []

        clauses: list[Clause] = []
        for section in _SECTIONS:
            section_pages = [
                p for p in pages if section.first_page <= p.page_number <= section.last_page
            ]
            if not section_pages:
                continue
            section_clauses = self._segment_section(section, section_pages)
            self.stats.per_section.append(
                (
                    section.label,
                    len(section_clauses),
                    self.stats.input_tokens,
                    self.stats.output_tokens,
                )
            )
            clauses.extend(section_clauses)
        return clauses

    def _segment_section(
        self, section: _Section, section_pages: Sequence[PageContent]
    ) -> list[Clause]:
        body = _format_pages_for_prompt(section_pages)
        user_message = (
            f"SECTION: {section.label}\n"
            f"STYLE HINT: {section.style_hint}\n\n"
            f"---BEGIN SECTION TEXT---\n{body}\n---END SECTION TEXT---"
        )

        logger.info(
            "Segmenting %r (pages %d-%d, %d chars).",
            section.label,
            section.first_page,
            section.last_page,
            len(body),
        )

        # Stream + per-section chunking keeps the connection warm during
        # long tool-call assembly; non-streaming requests at this size
        # intermittently 504 at the OpenRouter gateway.
        stream = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            stream=True,
            stream_options={"include_usage": True},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            tools=[_EMIT_CLAUSES_TOOL],
            tool_choice={"type": "function", "function": {"name": "emit_clauses"}},
        )

        tool_args, finish_reason = _drain_tool_call_stream(stream, stats=self.stats)
        self.stats.sections_called += 1

        if not tool_args:
            raise RuntimeError(
                f"{section.label}: no tool call emitted (finish_reason={finish_reason})."
            )

        try:
            payload = json.loads(tool_args)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{section.label}: tool arguments were not valid JSON: {exc}\n"
                f"first 500 chars: {tool_args[:500]}"
            ) from exc

        raw_clauses = payload.get("clauses")
        if not isinstance(raw_clauses, list):
            raise RuntimeError(
                f"{section.label}: emit_clauses payload was {type(raw_clauses).__name__}, "
                "expected list."
            )

        clauses: list[Clause] = []
        for entry in raw_clauses:
            try:
                clauses.append(Clause(**entry))
            except (TypeError, ValueError) as exc:
                logger.warning("%s: dropping malformed clause: %s (entry=%r)",
                               section.label, exc, entry)
        logger.info("%s: %d clauses.", section.label, len(clauses))
        return clauses


def _drain_tool_call_stream(
    stream: Any, *, stats: SegmentationStats
) -> tuple[str, str | None]:
    """Accumulate streamed tool-call argument deltas in index order."""
    args_by_index: dict[int, list[str]] = {}
    finish_reason: str | None = None

    for chunk in stream:
        if getattr(chunk, "usage", None) is not None:
            stats.input_tokens += chunk.usage.prompt_tokens or 0
            stats.output_tokens += chunk.usage.completion_tokens or 0
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        delta = choice.delta
        if delta is None or not delta.tool_calls:
            continue
        for tc in delta.tool_calls:
            idx = tc.index if tc.index is not None else 0
            if tc.function and tc.function.arguments:
                args_by_index.setdefault(idx, []).append(tc.function.arguments)

    if not args_by_index:
        return "", finish_reason
    first_idx = min(args_by_index)
    return "".join(args_by_index[first_idx]), finish_reason


def _format_pages_for_prompt(pages: Sequence[PageContent]) -> str:
    return "\n\n".join(f"[PAGE {p.page_number}]\n{p.text}" for p in pages)


def diagnostic_dump(pages: Sequence[PageContent]) -> str:
    """Return the prompt payload that would be sent to the LLM."""
    return _format_pages_for_prompt(pages)


__all__ = ["LLMSegmenter", "SegmentationStats", "diagnostic_dump"]
