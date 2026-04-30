"""End-to-end orchestration: PDF -> clauses -> JSON."""

from __future__ import annotations

import logging
from pathlib import Path

from charter_parser.models import ClauseDocument
from charter_parser.pdf_extract import extract_pages
from charter_parser.segment import LLMSegmenter

logger = logging.getLogger(__name__)


def run_pipeline(
    pdf_path: Path,
    *,
    first_page: int,
    last_page: int,
    segmenter: LLMSegmenter | None = None,
) -> ClauseDocument:
    logger.info("Extracting pages %d-%d from %s", first_page, last_page, pdf_path.name)
    pages = extract_pages(pdf_path, first_page=first_page, last_page=last_page)
    total_chars = sum(len(p.text) for p in pages)
    logger.info("Cleaned text: %d pages, %d chars.", len(pages), total_chars)

    active_segmenter = segmenter or LLMSegmenter()
    clauses = active_segmenter.segment(pages)
    s = active_segmenter.stats
    logger.info(
        "Segmenter: %d clauses, %d section call(s), input=%d output=%d tokens.",
        len(clauses), s.sections_called, s.input_tokens, s.output_tokens,
    )

    return ClauseDocument(
        source_pdf=pdf_path.name,
        page_range=(first_page, last_page),
        clauses=clauses,
    )
