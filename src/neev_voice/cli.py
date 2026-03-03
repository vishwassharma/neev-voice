"""CLI entry point for Neev Voice.

Provides commands for voice listening, document discussion,
configuration display, and provider listing using Typer and Rich.
Uses push-to-talk (hold SPACEBAR, release to pause, ENTER to send).
"""

import asyncio
import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from neev_voice.audio.keyboard import RecordingState
from neev_voice.audio.recorder import RecordingCancelledError
from neev_voice.config import (
    CONFIG_FILE,
    NeevSettings,
    SarvamSTTMode,
    STTProviderType,
    TTSProviderType,
    create_default_config,
    update_config_value,
)

app = typer.Typer(
    name="neev",
    help="Neev Voice - Hindi-English voice agent for intent extraction and document discussion.",
    rich_markup_mode="markdown",
)
console = Console()

# Config sub-app
config_app = typer.Typer(
    name="config",
    help="Manage Neev Voice configuration.",
    invoke_without_command=True,
)
app.add_typer(config_app, name="config")


def _get_settings() -> NeevSettings:
    """Load application settings from environment and JSON config.

    Loads settings from init kwargs, NEEV_* env vars, .env file,
    and ~/.config/neev/voice.json (if it exists). The JSON config
    file is not auto-created; use ``neev config init`` to create it.

    Returns:
        Loaded NeevSettings instance.
    """
    return NeevSettings()


@app.command()
def listen(
    stt: Optional[str] = typer.Option(None, "--stt", help="STT provider (sarvam)"),
    tts: Optional[str] = typer.Option(None, "--tts", help="TTS provider (sarvam, edge)"),
    mode: Optional[str] = typer.Option(
        None,
        "--mode",
        "-m",
        help="STT mode: translate (Hindi→English), codemix (mixed), formal (Devanagari)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Record audio with push-to-talk, transcribe, and extract intent.

    Hold **SPACEBAR** to record, release to pause, press **ENTER** to
    finalize, or **ESC** to cancel. Audio is transcribed and intent is extracted.
    """
    asyncio.run(_listen_async(stt, tts, mode, verbose))


def _make_recording_status(state: RecordingState) -> Panel:
    """Create a Rich Panel showing the current recording state.

    Args:
        state: Current push-to-talk recording state.

    Returns:
        Rich Panel with state-appropriate styling.
    """
    styles = {
        RecordingState.IDLE: ("Waiting... Hold SPACEBAR to record", "dim"),
        RecordingState.RECORDING: ("RECORDING  Hold SPACEBAR...", "bold red"),
        RecordingState.PAUSED: (
            "PAUSED  Hold SPACEBAR to continue, ENTER to send, ESC to cancel",
            "bold yellow",
        ),
        RecordingState.DONE: ("Captured! Processing...", "bold green"),
        RecordingState.CANCELLED: ("Cancelled.", "bold dim"),
    }
    text, style = styles.get(state, ("", ""))
    return Panel(Text(text, style=style), title="Push-to-Talk", border_style="cyan")


async def _listen_async(
    stt_name: str | None, tts_name: str | None, mode: str | None, verbose: bool
) -> None:
    """Async implementation of the listen command.

    Creates a scratch pad, records audio, transcribes, enriches via
    the enrichment agent, extracts intent, and saves all artifacts.

    Args:
        stt_name: Optional STT provider name override.
        tts_name: Optional TTS provider name override.
        mode: Optional STT mode override (translate, codemix, formal).
        verbose: Whether to show verbose output.
    """
    from neev_voice.audio.recorder import AudioRecorder
    from neev_voice.intent.extractor import IntentExtractor
    from neev_voice.llm.agent import EnrichmentAgent
    from neev_voice.scratch import ScratchPad
    from neev_voice.stt.sarvam import get_stt_provider

    settings = _get_settings()

    if mode:
        try:
            settings.stt_mode = SarvamSTTMode(mode)
        except ValueError:
            available = ", ".join(m.value for m in SarvamSTTMode)
            console.print(f"[red]Error:[/red] Unknown STT mode '{mode}'. Available: {available}")
            raise typer.Exit(1)

    provider_name = stt_name or settings.stt_provider.value

    try:
        stt_provider = get_stt_provider(provider_name, settings)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    scratch = ScratchPad("listen")
    recorder = AudioRecorder(settings=settings)
    agent = EnrichmentAgent(settings, scratch_path=str(scratch.flow_dir))
    extractor = IntentExtractor(agent)

    console.print(
        "[bold cyan]Push-to-Talk:[/bold cyan] "
        "Hold SPACEBAR to record, release to pause, ENTER to send, ESC to cancel"
    )

    live = Live(_make_recording_status(RecordingState.IDLE), console=console, refresh_per_second=4)

    def update_display(state: RecordingState) -> None:
        """Update Rich Live display with new recording state."""
        live.update(_make_recording_status(state))

    try:
        with live:
            segment = await recorder.record_push_to_talk(on_state_change=update_display)
    except RecordingCancelledError:
        console.print("[yellow]Recording cancelled.[/yellow]")
        raise typer.Exit(0)
    except RuntimeError as e:
        console.print(f"[red]Recording error:[/red] {e}")
        raise typer.Exit(1)

    if verbose:
        console.print(f"[dim]Recorded {segment.duration:.1f}s of audio[/dim]")

    wav_path = AudioRecorder.save_wav(segment)

    # Copy audio to scratch pad
    shutil.copy2(str(wav_path), str(scratch.audio_path))

    with console.status("[bold green]Transcribing..."):
        try:
            transcription = await stt_provider.transcribe(wav_path)
        except RuntimeError as e:
            console.print(f"[red]Transcription error:[/red] {e}")
            raise typer.Exit(1)

    scratch.save_transcription(transcription.text)
    console.print(Panel(transcription.text, title="Transcription", border_style="green"))

    if verbose:
        console.print(
            f"[dim]Language: {transcription.language} | "
            f"Confidence: {transcription.confidence:.2f} | "
            f"Provider: {transcription.provider}[/dim]"
        )

    with console.status("[bold blue]Extracting intent..."):
        try:
            intent = await extractor.extract(transcription.text)
        except RuntimeError as e:
            console.print(f"[red]Intent extraction error:[/red] {e}")
            raise typer.Exit(1)

    # Save enriched output if the agent produced content
    if hasattr(extractor, "_last_enriched") and extractor._last_enriched:
        scratch.save_enriched(extractor._last_enriched)

    scratch.save_metadata(
        transcription=transcription.text,
        intent_category=intent.category.value,
        intent_summary=intent.summary,
        duration=segment.duration,
    )

    _display_intent(intent)

    if verbose:
        console.print(f"[dim]Artifacts saved to: {scratch.flow_dir}[/dim]")


def _display_intent(intent: "ExtractedIntent") -> None:  # noqa: F821
    """Display extracted intent with Rich formatting.

    Args:
        intent: The extracted intent to display.
    """
    category_colors = {
        "problem_statement": "red",
        "solution": "green",
        "clue": "yellow",
        "mixed": "cyan",
        "agreement": "green",
        "disagreement": "red",
        "question": "blue",
    }

    color = category_colors.get(intent.category.value, "white")
    console.print(
        Panel(
            f"[bold {color}]{intent.category.value.upper()}[/bold {color}]\n\n"
            f"{intent.summary}\n\n"
            + (
                "[bold]Key Points:[/bold]\n" + "\n".join(f"  - {kp}" for kp in intent.key_points)
                if intent.key_points
                else ""
            ),
            title="Intent Analysis",
            border_style=color,
        )
    )


@app.command()
def discuss(
    document_path: str = typer.Argument(..., help="Path to document to discuss"),
    stt: Optional[str] = typer.Option(None, "--stt", help="STT provider"),
    tts: Optional[str] = typer.Option(None, "--tts", help="TTS provider"),
    mode: Optional[str] = typer.Option(
        None,
        "--mode",
        "-m",
        help="STT mode: translate (Hindi→English), codemix (mixed), formal (Devanagari)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Interactively discuss a document with voice input.

    Walks through each section of the document, reads it aloud
    via TTS, records the user's voice response, and classifies
    agreement or disagreement.
    """
    asyncio.run(_discuss_async(document_path, stt, tts, mode, verbose))


async def _discuss_async(
    document_path: str,
    stt_name: str | None,
    tts_name: str | None,
    mode: str | None,
    verbose: bool,
) -> None:
    """Async implementation of the discuss command.

    Creates a scratch pad, reads the latest listen folder for context,
    discusses the document section by section, and saves all artifacts.

    Args:
        document_path: Path to the document to discuss.
        stt_name: Optional STT provider name override.
        tts_name: Optional TTS provider name override.
        mode: Optional STT mode override (translate, codemix, formal).
        verbose: Whether to show verbose output.
    """
    from neev_voice.audio.recorder import AudioRecorder
    from neev_voice.discussion.manager import DiscussionManager
    from neev_voice.intent.extractor import IntentExtractor
    from neev_voice.llm.agent import EnrichmentAgent
    from neev_voice.scratch import ScratchPad
    from neev_voice.stt.sarvam import get_stt_provider
    from neev_voice.tts.edge import get_tts_provider

    settings = _get_settings()

    if mode:
        try:
            settings.stt_mode = SarvamSTTMode(mode)
        except ValueError:
            available = ", ".join(m.value for m in SarvamSTTMode)
            console.print(f"[red]Error:[/red] Unknown STT mode '{mode}'. Available: {available}")
            raise typer.Exit(1)

    stt_provider_name = stt_name or settings.stt_provider.value
    tts_provider_name = tts_name or settings.tts_provider.value

    try:
        stt_provider = get_stt_provider(stt_provider_name, settings)
        tts_provider = get_tts_provider(tts_provider_name, settings)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    scratch = ScratchPad("discussion")

    # Read latest listen folder for context (reserved for future agent integration)
    latest_listen = ScratchPad.get_latest_folder("listen")
    _context = None
    if latest_listen:
        transcription_file = latest_listen / "transcription.txt"
        if transcription_file.exists():
            _context = transcription_file.read_text(encoding="utf-8")

    recorder = AudioRecorder(settings=settings)
    agent = EnrichmentAgent(settings, scratch_path=str(scratch.flow_dir))
    extractor = IntentExtractor(agent)

    manager = DiscussionManager(
        settings=settings,
        recorder=recorder,
        stt=stt_provider,
        tts=tts_provider,
        intent_extractor=extractor,
    )

    doc_path = Path(document_path)
    console.print(f"[bold]Discussing:[/bold] {doc_path}")

    try:
        results = await manager.run_discussion(doc_path)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    # Save section results to scratch pad
    for i, result in enumerate(results, 1):
        scratch.save_section(
            i,
            {
                "section": result.section,
                "user_response": result.user_response,
                "intent": result.intent.value,
                "summary": result.summary,
            },
        )

    agreements = sum(1 for r in results if r.intent.value == "agreement")
    disagreements = sum(1 for r in results if r.intent.value == "disagreement")

    scratch.save_summary(
        {
            "total_sections": len(results),
            "agreements": agreements,
            "disagreements": disagreements,
            "document_path": str(doc_path),
        }
    )

    scratch.save_metadata(
        document_path=str(doc_path),
        total_sections=len(results),
        agreements=agreements,
        disagreements=disagreements,
    )

    # Generate discussion-result.md with consolidated insights
    scratch.save_discussion_result(
        _build_discussion_result_md(doc_path, results, agreements, disagreements)
    )

    console.print(f"\n[bold]Discussion complete![/bold] {len(results)} sections reviewed.")

    table = Table(title="Discussion Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")
    table.add_row("Total Sections", str(len(results)))
    table.add_row("Agreements", f"[green]{agreements}[/green]")
    table.add_row("Disagreements", f"[red]{disagreements}[/red]")
    console.print(table)

    if verbose:
        console.print(f"[dim]Artifacts saved to: {scratch.flow_dir}[/dim]")


def _build_discussion_result_md(
    doc_path: Path,
    results: list,
    agreements: int,
    disagreements: int,
) -> str:
    """Build a discussion-result.md markdown document.

    Consolidates all section results into a single markdown file
    with insights, agreements, disagreements, and key takeaways.

    Args:
        doc_path: Path to the discussed document.
        results: List of DiscussionResult objects.
        agreements: Total agreement count.
        disagreements: Total disagreement count.

    Returns:
        Markdown string for discussion-result.md.
    """
    lines = [
        f"# Discussion Result: {doc_path.name}",
        "",
        f"**Document:** {doc_path}",
        f"**Sections reviewed:** {len(results)}",
        f"**Agreements:** {agreements} | **Disagreements:** {disagreements}",
        "",
        "---",
        "",
    ]

    # Collect by category
    agreement_results = [r for r in results if r.intent.value == "agreement"]
    disagreement_results = [r for r in results if r.intent.value == "disagreement"]
    other_results = [r for r in results if r.intent.value not in ("agreement", "disagreement")]

    if agreement_results:
        lines.append("## Agreements")
        lines.append("")
        for r in agreement_results:
            section_title = r.section.split("\n")[0].strip()
            lines.append(f"- **{section_title}**: {r.summary}")
        lines.append("")

    if disagreement_results:
        lines.append("## Disagreements")
        lines.append("")
        for r in disagreement_results:
            section_title = r.section.split("\n")[0].strip()
            lines.append(f"- **{section_title}**: {r.summary}")
        lines.append("")

    if other_results:
        lines.append("## Explorations & Insights")
        lines.append("")
        for r in other_results:
            section_title = r.section.split("\n")[0].strip()
            lines.append(f"- **{section_title}** ({r.intent.value}): {r.summary}")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Section Details")
    lines.append("")

    for i, r in enumerate(results, 1):
        section_title = r.section.split("\n")[0].strip()
        lines.append(f"### {i}. {section_title}")
        lines.append("")
        lines.append(f"**Intent:** {r.intent.value}")
        lines.append(f"**User response:** {r.user_response}")
        lines.append(f"**Summary:** {r.summary}")
        lines.append("")

    return "\n".join(lines)


@config_app.callback()
def config_show(ctx: typer.Context) -> None:
    """Show current configuration settings."""
    if ctx.invoked_subcommand is not None:
        return

    settings = _get_settings()

    table = Table(title="Neev Voice Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("STT Provider", settings.stt_provider.value)
    table.add_row("STT Mode", settings.stt_mode.value)
    table.add_row("TTS Provider", settings.tts_provider.value)
    table.add_row("Sample Rate", f"{settings.sample_rate} Hz")
    table.add_row("Silence Threshold", str(settings.silence_threshold))
    table.add_row("Silence Duration", f"{settings.silence_duration}s")
    table.add_row("Key Release Timeout", f"{settings.key_release_timeout}s")
    table.add_row("Claude Model", settings.claude_model)
    table.add_row("Claude Timeout", f"{settings.claude_timeout}s")
    table.add_row("LLM Provider", settings.llm_provider.value)
    table.add_row("LLM API Base", settings.llm_api_base or "(provider default)")
    table.add_row("STT Max Audio Duration", f"{settings.stt_max_audio_duration}s")
    table.add_row(
        "Sarvam API Key",
        "[green]configured[/green]" if settings.sarvam_api_key else "[red]not set[/red]",
    )
    table.add_row(
        "Anthropic API Key",
        "[green]configured[/green]" if settings.anthropic_api_key else "[red]not set[/red]",
    )
    table.add_row(
        "OpenRouter API Key",
        "[green]configured[/green]" if settings.openrouter_api_key else "[red]not set[/red]",
    )

    console.print(table)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Setting name (e.g. stt_mode, claude_model)"),
    value: str = typer.Argument(..., help="Value to set"),
) -> None:
    """Update a persistent config value in ~/.config/neev/voice.json."""
    try:
        update_config_value(key, value)
        console.print(f"[green]Set[/green] {key} = {value}")
    except KeyError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@config_app.command("init")
def config_init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config file"),
) -> None:
    """Create a default config file at ~/.config/neev/voice.json.

    Writes all non-secret settings with their default values.
    API keys are excluded and must be set via environment variables.
    """
    try:
        path = create_default_config(force=force)
        console.print(f"[green]Created config file:[/green] {path}")
    except FileExistsError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@config_app.command("path")
def config_path() -> None:
    """Print the path to the persistent config file."""
    console.print(str(CONFIG_FILE))


@app.command()
def version() -> None:
    """Display the current Neev Voice version."""
    from neev_voice import __version__

    console.print(f"neev-voice {__version__}")


@app.command()
def providers() -> None:
    """List available STT and TTS providers."""
    table = Table(title="Available Providers")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Notes")

    for p in STTProviderType:
        notes = "Requires NEEV_SARVAM_API_KEY" if p == STTProviderType.SARVAM else ""
        table.add_row("STT", p.value, notes)

    for p in TTSProviderType:
        if p == TTSProviderType.SARVAM:
            notes = "Requires NEEV_SARVAM_API_KEY"
        elif p == TTSProviderType.EDGE:
            notes = "Free, no API key needed"
        else:
            notes = ""
        table.add_row("TTS", p.value, notes)

    console.print(table)


if __name__ == "__main__":
    app()
