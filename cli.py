"""CLI entry point for late-edition using Typer."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

import config
from chunk import split_into_chunks
from config import GEMINI_API_KEY
from correct import correct_chunk_with_gemini, mock_correct_chunk
from fetch import fetch_loc_page, fetch_offline
from guard import validate_chunk
from report import generate_report

app = typer.Typer(
    name="late-edition",
    help="Late Edition: Public-domain historic newspaper OCR correction with Gemma 4 and safety guards.",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    date: Optional[str] = typer.Option(
        None,
        "--date",
        "-d",
        help="Date of newspaper issue in YYYY-MM-DD format (e.g. 1900-01-01).",
    ),
    offline: Optional[Path] = typer.Option(
        None,
        "--offline",
        help="Path to local folder containing page image and raw OCR .txt file (for example: sample_data/).",
    ),
    newspaper: Optional[str] = typer.Option(
        None,
        "--newspaper",
        "-n",
        help="Optional Chronicling America LCCN identifier (e.g. sn83030214).",
    ),
    output: Path = typer.Option(
        Path("report.html"),
        "--output",
        "-o",
        help="Destination path for generated static HTML report.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Gemini / Gemma model name (defaults to GEMMA_MODEL in .env).",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Run without calling Gemini API (simulates corrections for testing).",
    ),
) -> None:
    """Fetch historic newspaper page, correct OCR text with Gemma 4, validate with guards, and output HTML report."""
    console.print("[bold yellow]=== Late Edition ===[/bold yellow]")
    console.print("Historic Newspaper OCR Correction & Guard System\n")

    # 1. Validate input source
    if not date and not offline:
        console.print(
            "[bold red]Error:[/bold red] You must provide either [bold]--date YYYY-MM-DD[/bold] "
            "to fetch from Chronicling America, or [bold]--offline <folder>[/bold] for local mode."
        )
        raise typer.Exit(code=1)

    # 2. Acquire page image + raw OCR text
    try:
        if offline:
            console.print(f"[cyan]Loading from offline folder:[/cyan] {offline}")
            fetch_result = fetch_offline(offline)
        else:
            console.print(f"[cyan]Fetching from Chronicling America (LOC):[/cyan] Date={date}, Newspaper={newspaper or 'Any'}")
            fetch_result = fetch_loc_page(date=date, newspaper=newspaper)
    except Exception as exc:
        console.print(f"[bold red]Fetch Failed:[/bold red] {exc}")
        if not offline:
            console.print(
                "\n[yellow]Tip:[/yellow] If network access to the Library of Congress API is unavailable, "
                "you can run in manual offline mode:\n"
                "  late-edition run --offline <folder>\n"
            )
        raise typer.Exit(code=1)

    console.print(
        f"[green]Acquired page:[/green] [bold]{fetch_result.newspaper_title}[/bold] "
        f"({fetch_result.date}, LCCN: {fetch_result.lccn})"
    )
    console.print(f"Image: {fetch_result.image_path} | Raw OCR: {len(fetch_result.ocr_text)} chars\n")

    # 3. Chunk text into ~800 character blocks on paragraph/sentence boundaries
    chunks = split_into_chunks(fetch_result.ocr_text, target_size=800)
    console.print(f"Split OCR text into [bold]{len(chunks)}[/bold] chunk(s) of ~800 characters.\n")

    if not chunks:
        console.print("[yellow]Warning: OCR text was empty. Generating empty report.[/yellow]")
        generate_report(
            newspaper_title=fetch_result.newspaper_title,
            date=fetch_result.date,
            lccn=fetch_result.lccn,
            sequence=fetch_result.sequence,
            image_path=fetch_result.image_path,
            raw_ocr_text="",
            guard_results=[],
            output_path=output,
            model_name=model,
        )
        return

    # Resolve model name
    try:
        active_model = model or config.DEFAULT_MODEL
    except ValueError as e:
        if not mock:
            console.print(f"[bold red]Error:[/bold red] {e}")
            raise typer.Exit(code=1)
        active_model = "mock"

    # Check API key if not in mock mode
    if not mock and not GEMINI_API_KEY:
        console.print(
            "[bold red]Error:[/bold red] GEMINI_API_KEY is not set.\n"
            "Please copy [bold].env.example[/bold] to [bold].env[/bold] and set your Gemini API key,\n"
            "or run with [bold]--mock[/bold] for offline testing without an API key."
        )
        raise typer.Exit(code=1)

    # 4. Correct each chunk with Gemma 4 and validate with guards
    guard_results = []
    console.print(f"[cyan]Processing chunks with model: {active_model} (Mock: {mock})[/cyan]")

    for idx, chunk in enumerate(chunks, start=1):
        console.print(f"  [bold]Chunk {idx}/{len(chunks)}[/bold] ({len(chunk)} chars)... ", end="")
        try:
            if mock:
                corrected = mock_correct_chunk(chunk)
            else:
                corrected = correct_chunk_with_gemini(chunk, model_name=active_model)
        except Exception as exc:
            console.print(f"[red]Correction error:[/red] {exc}. Reverting to original chunk.")
            corrected = chunk

        # Validate with difflib guard
        guard_res = validate_chunk(original=chunk, corrected=corrected)
        guard_results.append(guard_res)

        if guard_res.accepted:
            edit_count = len(guard_res.hunks)
            console.print(
                f"[bold green]ACCEPTED[/bold green] "
                f"({guard_res.similarity_ratio * 100:.1f}% sim, {edit_count} edits)"
            )
        else:
            reasons = "; ".join(guard_res.flag_reasons)
            console.print(
                f"[bold red]REVERTED[/bold red] "
                f"({guard_res.similarity_ratio * 100:.1f}% sim) - [italic]{reasons}[/italic]"
            )
            if guard_res.hauntings:
                console.print(f"    [magenta]Hauntings detected:[/magenta] {', '.join(guard_res.hauntings)}")

    # 5. Generate HTML Report
    console.print(f"\n[cyan]Generating HTML report:[/cyan] {output}")
    report_file = generate_report(
        newspaper_title=fetch_result.newspaper_title,
        date=fetch_result.date,
        lccn=fetch_result.lccn,
        sequence=fetch_result.sequence,
        image_path=fetch_result.image_path,
        raw_ocr_text=fetch_result.ocr_text,
        guard_results=guard_results,
        output_path=output,
        model_name=model,
    )

    accepted_count = sum(1 for r in guard_results if r.accepted)
    reverted_count = len(guard_results) - accepted_count
    hauntings_count = sum(len(r.hauntings) for r in guard_results)

    console.print("\n[bold green]Report generated successfully![/bold green]")
    console.print(f"  Destination: [bold]{report_file.resolve()}[/bold]")
    console.print(f"  Chunks Accepted: {accepted_count} | Reverted: {reverted_count}")
    console.print(f"  Hauntings Blocked: {hauntings_count}")


@app.command(name="version")
def version() -> None:
    """Display late-edition version."""
    console.print("late-edition v0.1.0")


def main() -> None:
    """Entry point for command line execution."""
    app()


if __name__ == "__main__":
    main()
