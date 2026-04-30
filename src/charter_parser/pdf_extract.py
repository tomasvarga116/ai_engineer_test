"""PDF text extraction with strikethrough and gutter-number filtering.

Operates at the character layer (via pdfplumber) so we can drop glyphs
crossed by a horizontal rule. The downstream LLM segmenter sees only
clean, intended text.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pdfplumber
from pdfplumber.page import Page

logger = logging.getLogger(__name__)

_GUTTER_NUMBER_RE = re.compile(r"^\s*\d{1,3}(?:\s*[,;]\s*\d{1,3})*\s*$")
_TRAILING_GUTTER_NUMBER_RE = re.compile(r"\s+(?<![\d.,/$%])\d{1,3}\s*$")

# Strike detection tolerances (PDF points). A strike must hit close to the
# x-height midline and span at least half the glyph width.
_STRIKE_VERTICAL_TOLERANCE = 2.0
_STRIKE_MIN_HORIZONTAL_COVERAGE = 0.5


@dataclass(frozen=True)
class Line:
    text: str
    size: float
    page: int
    top: float


@dataclass
class PageContent:
    page_number: int
    lines: list[Line] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


def extract_pages(
    pdf_path: Path,
    *,
    first_page: int,
    last_page: int,
) -> list[PageContent]:
    """Extract cleaned text from a 1-indexed inclusive page range."""
    if first_page < 1 or last_page < first_page:
        raise ValueError(
            f"Invalid page range: first_page={first_page}, last_page={last_page}"
        )
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    pages: list[PageContent] = []
    with pdfplumber.open(pdf_path) as pdf:
        if last_page > len(pdf.pages):
            raise ValueError(
                f"PDF has {len(pdf.pages)} pages; last_page={last_page} is out of range."
            )
        for page_idx in range(first_page - 1, last_page):
            page = pdf.pages[page_idx]
            pages.append(_extract_page(page))
            logger.debug("page %d: %d lines", page_idx + 1, len(pages[-1].lines))
    return pages


def _extract_page(page: Page) -> PageContent:
    chars = page.chars
    if not chars:
        return PageContent(page_number=page.page_number)

    struck_ids = _strikethrough_char_ids(page)
    visible_chars = [c for i, c in enumerate(chars) if i not in struck_ids]
    if struck_ids:
        logger.debug(
            "page %d: removed %d/%d strikethrough chars",
            page.page_number, len(struck_ids), len(chars),
        )

    lines = _group_chars_into_lines(visible_chars, page.page_number)
    lines = [ln for ln in lines if not _GUTTER_NUMBER_RE.match(ln.text)]
    lines = [_strip_trailing_gutter_number(ln) for ln in lines]
    return PageContent(page_number=page.page_number, lines=lines)


def _strip_trailing_gutter_number(line: Line) -> Line:
    new_text = _TRAILING_GUTTER_NUMBER_RE.sub("", line.text)
    if new_text == line.text:
        return line
    return Line(text=new_text, size=line.size, page=line.page, top=line.top)


def _strikethrough_char_ids(page: Page) -> set[int]:
    """Return char indices crossed by a thin horizontal rule.

    Some PDF producers emit strikes as ``page.lines``, others as thin
    ``page.rects``. Treat both as candidates if they are roughly horizontal
    and very thin.
    """
    horizontals: list[tuple[float, float, float]] = []  # (y, x0, x1)

    for ln in page.lines:
        if abs(ln.get("height", 0)) <= 1.5 and ln.get("x1", 0) > ln.get("x0", 0):
            y = (ln["top"] + ln["bottom"]) / 2.0
            horizontals.append((y, ln["x0"], ln["x1"]))

    for r in page.rects:
        h = r.get("height", 0) or 0
        if 0 < h <= 1.5 and r["x1"] > r["x0"]:
            y = (r["top"] + r["bottom"]) / 2.0
            horizontals.append((y, r["x0"], r["x1"]))

    if not horizontals:
        return set()

    struck: set[int] = set()
    for idx, ch in enumerate(page.chars):
        midline = (ch["top"] + ch["bottom"]) / 2.0
        c_x0, c_x1 = ch["x0"], ch["x1"]
        c_width = max(c_x1 - c_x0, 0.1)
        for y, x0, x1 in horizontals:
            if abs(y - midline) > _STRIKE_VERTICAL_TOLERANCE:
                continue
            overlap = max(0.0, min(c_x1, x1) - max(c_x0, x0))
            if overlap / c_width >= _STRIKE_MIN_HORIZONTAL_COVERAGE:
                struck.add(idx)
                break
    return struck


def _group_chars_into_lines(chars: Iterable[dict], page_number: int) -> list[Line]:
    sorted_chars = sorted(chars, key=lambda c: (round(c["top"], 1), c["x0"]))
    lines: list[Line] = []
    current: list[dict] = []
    current_top: float | None = None
    line_tolerance = 2.5

    def flush() -> None:
        if not current:
            return
        text = _assemble_line_text(current)
        if not text.strip():
            return
        sizes = [c.get("size", 0) for c in current if c.get("size")]
        size = max(sizes) if sizes else 0.0
        top = min(c["top"] for c in current)
        lines.append(Line(text=text, size=size, page=page_number, top=top))

    for ch in sorted_chars:
        if current_top is None or abs(ch["top"] - current_top) <= line_tolerance:
            current.append(ch)
            current_top = ch["top"] if current_top is None else current_top
        else:
            flush()
            current = [ch]
            current_top = ch["top"]
    flush()
    return lines


def _assemble_line_text(line_chars: list[dict]) -> str:
    line_chars = sorted(line_chars, key=lambda c: c["x0"])
    out: list[str] = []
    prev_x1: float | None = None
    prev_width: float | None = None
    for ch in line_chars:
        text = ch.get("text", "")
        if not text:
            continue
        if prev_x1 is not None and prev_width is not None:
            gap = ch["x0"] - prev_x1
            if gap > prev_width * 0.3 and not (out and out[-1].endswith(" ")):
                out.append(" ")
        out.append(text)
        prev_x1 = ch["x1"]
        prev_width = max(ch["x1"] - ch["x0"], 0.1)
    return "".join(out).rstrip()
