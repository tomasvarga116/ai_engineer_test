"""Command-line entrypoint."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

from charter_parser.pdf_extract import extract_pages
from charter_parser.pipeline import run_pipeline
from charter_parser.segment import diagnostic_dump

app = typer.Typer(
    add_completion=False,
    help="Extract numbered legal clauses from a maritime charter party PDF.",
)
console = Console()


@app.command()
def extract(
    pdf: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True, help="Path to the PDF."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Where to write the JSON."),
    ] = Path("output/clauses.json"),
    first_page: Annotated[
        int,
        typer.Option("--first-page", help="First page of Part II (1-indexed, inclusive)."),
    ] = 6,
    last_page: Annotated[
        int,
        typer.Option("--last-page", help="Last page of Part II (1-indexed, inclusive)."),
    ] = 39,
    dump_prompt: Annotated[
        bool,
        typer.Option(
            "--dump-prompt",
            help="Skip the LLM call and write the cleaned input that would be sent.",
        ),
    ] = False,
    log_level: Annotated[
        str,
        typer.Option("--log-level", help="DEBUG, INFO, WARNING, ERROR."),
    ] = os.getenv("LOG_LEVEL", "INFO"),
) -> None:
    """Extract clauses from PDF and write them to JSON."""
    load_dotenv()
    _configure_logging(log_level)

    if dump_prompt:
        pages = extract_pages(pdf, first_page=first_page, last_page=last_page)
        output.parent.mkdir(parents=True, exist_ok=True)
        dump_path = output.with_suffix(".prompt.txt")
        dump_path.write_text(diagnostic_dump(pages), encoding="utf-8")
        console.print(f"Wrote {len(pages)} pages of cleaned text to [cyan]{dump_path}[/cyan]")
        return

    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[bold red]OPENAI_API_KEY is not set.[/bold red] "
            "Set it in .env, export it, or rerun with --dump-prompt."
        )
        raise typer.Exit(code=2)

    document = run_pipeline(pdf_path=pdf, first_page=first_page, last_page=last_page)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document.to_json(), encoding="utf-8")

    console.print(f"Wrote {len(document.clauses)} clauses to [cyan]{output}[/cyan]")


def _configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
    )


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
