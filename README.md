# Charter Party Clause Parser

Extracts numbered legal clauses from **Part II** of a maritime voyage
charter party PDF (`voyage-charter-example.pdf`) into structured JSON.

The committed deliverable is [`output/clauses.json`](./output/clauses.json)
(94 clauses across the three sub-sections of Part II).

The original task brief is in [`TASK.md`](./TASK.md).

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate              # Windows
# or:  source .venv/bin/activate    # macOS / Linux
pip install -e ".[dev]"

cp .env.example .env                # then paste your OpenRouter key into .env
```

`.env`:

```
OPENAI_API_KEY=sk-or-v1-...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_MODEL=google/gemini-2.5-flash
```

The OpenAI Python SDK works against any OpenAI-compatible endpoint, so
swapping providers (OpenAI proper, Together, Groq, self-hosted) is a
two-env-var change with no code changes.

## Usage

```bash
# Full pipeline, default page range 6-39:
python -m charter_parser ./voyage-charter-example.pdf

# Custom range / output path:
python -m charter_parser ./voyage-charter-example.pdf \
    --first-page 6 --last-page 39 \
    --output ./output/clauses.json

# No API key handy? Inspect what would be sent to the model:
python -m charter_parser ./voyage-charter-example.pdf --dump-prompt
# -> output/clauses.prompt.txt
```

## Tests

```bash
pytest -q
```

14 unit tests, no API calls — the OpenAI client is mocked.

## Architecture

Two stages. Each does the work the other can't.

```
PDF  --pdfplumber-->  cleaned PageContent  --LLM (per section)-->  Clause[]  --pydantic-->  JSON
```

**`pdf_extract.py`** — pdfplumber-based. Removes strikethrough glyphs at
the character layer (find horizontal rules in `page.lines` / `page.rects`,
mark any glyph whose midline overlaps), strips bare line numbers from the
gutter, and groups characters into ordered lines. The LLM never sees ink,
so it cannot detect strikethroughs — this layer must run first.

**`segment.py`** — LLM-driven. The document has three different visual
conventions for clause headings:

| Pages | Section | Heading style |
|---|---|---|
| 6–17 | SHELLVOY 5 base form | Two-column layout. Titles in the **left margin gutter** (`Condition` / `Of vessel`), body to the right. All 9pt — no font signal. |
| 18–35 | Shell additional clauses | Title-case headings on their own line (`1. Indemnity Clause`). All 12pt — no font signal. |
| 36–39 | Charterers' additional clauses | ALL-CAPS headings (`1. INTERNATIONAL REGULATIONS CLAUSE`). Numbering restarts at 1. |

A single regex / font-size segmenter doesn't cover all three. The LLM is
called once per section with a section-specific style hint, the response
is forced through a function call (`emit_clauses`) for guaranteed-parseable
structured output, and the response is streamed (the gateway times out on
non-streamed tool calls of this size).

**`models.py`** — pydantic. `Clause(id, title, text)` validates every
entry; malformed entries from the LLM are dropped with a warning.

**`pipeline.py` / `cli.py`** — orchestration and the Typer CLI.

## Notes on the output

* **One spurious row** in the committed JSON: section-1 clause `id=39`
  has title `"3)"` — the LLM promoted a sub-bullet from clause 38's body.
  Documented rather than scrubbed; tightening the section-1 style hint
  with a few-shot example fixes it.
* Section 3 contains a few **self-titled clauses** where the PDF format
  is `"<n>. <full sentence>"` with no separate heading; in those rows
  `title == text`. That mirrors the source.
* Several clauses have body `"Not Applicable"` — those are real entries
  in SHELLVOY 5 that the parties marked as not used. Preserved verbatim.

## Security

`.env` is gitignored; only `.env.example` is committed. The CLI exits
with an error if `OPENAI_API_KEY` is unset (unless `--dump-prompt` is
passed).
