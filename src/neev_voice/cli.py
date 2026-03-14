"""CLI entry point for Neev Voice.

Provides commands for voice listening, text enrichment, document discussion,
configuration display, and provider listing using Typer and Rich.
Uses push-to-talk (hold SPACEBAR, release to pause, ENTER to send).
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import click
import structlog
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
    EnrichmentVersion,
    NeevSettings,
    SarvamSTTMode,
    STTProviderType,
    TTSProviderType,
    create_default_config,
    update_config_value,
)
from neev_voice.exceptions import NeevConfigError, NeevError, TranscriptRejectedError
from neev_voice.log import configure_logging

if TYPE_CHECKING:
    from neev_voice.intent.extractor import ExtractedIntent
    from neev_voice.llm.agent import EnrichmentAgent
    from neev_voice.llm.enrichment_loop import EnrichmentLoopAgent

logger = structlog.get_logger(__name__)

__all__ = ["app", "config_app"]

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


def _get_enrichment_agent(
    settings: NeevSettings, scratch_path: str
) -> EnrichmentAgent | EnrichmentLoopAgent:
    """Create an enrichment agent based on the configured version.

    Returns an EnrichmentAgent (v1) for single-pass enrichment, or
    an EnrichmentLoopAgent (v2) for iterative Ralph Loop enrichment.

    Args:
        settings: Application settings with enrichment_version.
        scratch_path: Path to the scratch pad flow directory.

    Returns:
        An enrichment agent instance (v1 or v2).
    """
    from neev_voice.llm.agent import EnrichmentAgent
    from neev_voice.llm.enrichment_loop import EnrichmentLoopAgent

    if settings.enrichment_version == EnrichmentVersion.V2:
        return EnrichmentLoopAgent(
            settings,
            scratch_path,
            max_iterations=settings.enrichment_max_iterations,
        )
    return EnrichmentAgent(settings, scratch_path=scratch_path)


@app.command()
def listen(
    stt: str | None = typer.Option(None, "--stt", help="STT provider (sarvam)"),
    tts: str | None = typer.Option(None, "--tts", help="TTS provider (sarvam, edge)"),
    mode: str | None = typer.Option(
        None,
        "--mode",
        "-m",
        help="STT mode: translate (Hindi→English), codemix (mixed), formal (Devanagari)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    no_review: bool = typer.Option(False, "--no-review", help="Skip transcript review step"),
) -> None:
    """Record audio with push-to-talk, transcribe, and extract intent.

    Hold **SPACEBAR** to record, release to pause, press **ENTER** to
    finalize, or **ESC** to cancel. Audio is transcribed and intent is extracted.
    """
    asyncio.run(_listen_async(stt, tts, mode, verbose, no_review=no_review))


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
    stt_name: str | None,
    tts_name: str | None,
    mode: str | None,
    verbose: bool,
    *,
    no_review: bool = False,
) -> None:
    """Async implementation of the listen command.

    Creates a scratch pad, records audio, transcribes, enriches via
    the enrichment agent, extracts intent, and saves all artifacts.

    Args:
        stt_name: Optional STT provider name override.
        tts_name: Optional TTS provider name override.
        mode: Optional STT mode override (translate, codemix, formal).
        verbose: Whether to show verbose output.
        no_review: If True, skip the transcript review gate.
    """
    from neev_voice.audio.recorder import AudioRecorder
    from neev_voice.intent.classifier import IntentClassifier
    from neev_voice.scratch import ScratchPad
    from neev_voice.stt.sarvam import get_stt_provider

    configure_logging(json_logs=not verbose)
    settings = _get_settings()
    logger.info("listen_command_started")

    if mode:
        try:
            settings.stt_mode = SarvamSTTMode(mode)
        except ValueError:
            available = ", ".join(m.value for m in SarvamSTTMode)
            console.print(f"[red]Error:[/red] Unknown STT mode '{mode}'. Available: {available}")
            raise typer.Exit(1) from None

    provider_name = stt_name or settings.stt_provider.value

    try:
        stt_provider = get_stt_provider(provider_name, settings)
    except (ValueError, NeevConfigError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None

    scratch = ScratchPad("listen")
    recorder = AudioRecorder(settings=settings)
    agent = _get_enrichment_agent(settings, str(scratch.flow_dir))
    classifier = IntentClassifier(settings)

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
        raise typer.Exit(0) from None
    except RuntimeError as e:
        console.print(f"[red]Recording error:[/red] {e}")
        raise typer.Exit(1) from None

    if verbose:
        console.print(f"[dim]Recorded {segment.duration:.1f}s of audio[/dim]")

    wav_path = AudioRecorder.save_wav(segment)

    try:
        # Copy audio to scratch pad
        shutil.copy2(str(wav_path), str(scratch.audio_path))

        with console.status("[bold green]Transcribing..."):
            try:
                transcription = await stt_provider.transcribe(wav_path)
            except (RuntimeError, NeevError) as e:
                console.print(f"[red]Transcription error:[/red] {e}")
                raise typer.Exit(1) from None
    finally:
        wav_path.unlink(missing_ok=True)

    scratch.save_transcription(transcription.text)

    if verbose:
        console.print(
            f"[dim]Language: {transcription.language} | "
            f"Confidence: {transcription.confidence:.2f} | "
            f"Provider: {transcription.provider}[/dim]"
        )

    # Transcript review gate
    transcript_text = transcription.text
    if no_review:
        console.print(Panel(transcript_text, title="Transcription", border_style="green"))
    else:
        from neev_voice.review import TranscriptReviewer

        reviewer = TranscriptReviewer(console=console)
        try:
            action, transcript_text = await reviewer.review(
                transcript_text, scratch.transcription_path
            )
            if action.value == "edit":
                scratch.save_transcription(transcript_text)
        except TranscriptRejectedError:
            console.print("[yellow]Transcript rejected.[/yellow]")
            raise typer.Exit(0) from None

    with console.status("[bold blue]Enriching..."):
        try:
            enriched = await agent.enrich(transcript_text)
        except (RuntimeError, NeevError) as e:
            console.print(f"[red]Enrichment error:[/red] {e}")
            raise typer.Exit(1) from None

    scratch.save_enriched(enriched)
    if enriched:
        console.print(Panel(enriched, title="Enrichment", border_style="blue"))

    with console.status("[bold blue]Classifying intent..."):
        try:
            intent = await classifier.classify(transcript_text)
        except (RuntimeError, NeevError) as e:
            console.print(f"[red]Intent classification error:[/red] {e}")
            raise typer.Exit(1) from None

    scratch.save_metadata(
        transcription=transcript_text,
        intent_category=intent.category.value,
        intent_summary=intent.summary,
        duration=segment.duration,
    )

    _display_intent(intent)

    if verbose:
        console.print(f"[dim]Artifacts saved to: {scratch.flow_dir}[/dim]")


def _display_intent(intent: ExtractedIntent) -> None:
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
def enrich(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Enter text in your default editor for enrichment and intent extraction.

    Opens **$EDITOR** (or system default) so you can type or paste text.
    The text is then enriched and classified, identical to the post-transcription
    flow in ``neev listen``.
    """
    asyncio.run(_enrich_async(verbose))


async def _enrich_async(verbose: bool) -> None:
    """Async implementation of the enrich command.

    Opens the user's default editor, reads the entered text, enriches it
    via the configured enrichment agent, and classifies intent.

    Args:
        verbose: Whether to show verbose output.
    """
    from neev_voice.intent.classifier import IntentClassifier
    from neev_voice.scratch import ScratchPad

    configure_logging(json_logs=not verbose)
    settings = _get_settings()
    logger.info("enrich_command_started")

    text = click.edit(text="", extension=".txt")
    if not text or not text.strip():
        console.print("[yellow]No text entered. Aborting.[/yellow]")
        raise typer.Exit(0)

    text = text.strip()

    scratch = ScratchPad("enrich")
    agent = _get_enrichment_agent(settings, str(scratch.flow_dir))
    classifier = IntentClassifier(settings)

    scratch.save_transcription(text)
    console.print(Panel(text, title="Input Text", border_style="green"))

    with console.status("[bold blue]Enriching..."):
        try:
            enriched = await agent.enrich(text)
        except (RuntimeError, NeevError) as e:
            console.print(f"[red]Enrichment error:[/red] {e}")
            raise typer.Exit(1) from None

    scratch.save_enriched(enriched)
    if enriched:
        console.print(Panel(enriched, title="Enrichment", border_style="blue"))

    with console.status("[bold blue]Classifying intent..."):
        try:
            intent = await classifier.classify(text)
        except (RuntimeError, NeevError) as e:
            console.print(f"[red]Intent classification error:[/red] {e}")
            raise typer.Exit(1) from None

    scratch.save_metadata(
        transcription=text,
        intent_category=intent.category.value,
        intent_summary=intent.summary,
    )

    _display_intent(intent)

    if verbose:
        console.print(f"[dim]Artifacts saved to: {scratch.flow_dir}[/dim]")


@app.command()
def discuss(
    name: str | None = typer.Option(None, "-n", help="Session name (auto-generated if omitted)"),
    files: str | None = typer.Option(
        None, "--files", help="Research folder path (required for new)"
    ),
    source: str | None = typer.Option(
        None, "--source", help="Source code root (default: git root)"
    ),
    continue_session: bool = typer.Option(False, "--continue", help="Continue last session"),
    resume: str | None = typer.Option(None, "--resume", help="Resume a specific session by name"),
    reset: bool = typer.Option(False, "--reset", help="Reset session state to prepare"),
    enquery: bool = typer.Option(False, "--enquery", help="Jump to enquiry state"),
    output: str | None = typer.Option(
        None, "--output", help="Output folder (default: scratch pad)"
    ),
    stt: str | None = typer.Option(None, "--stt", help="STT provider"),
    tts: str | None = typer.Option(None, "--tts", help="TTS provider"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Interactive research discussion with state machine.

    Analyzes research documents, generates progressive tutorials,
    and presents them via TTS with interactive enquiry support.

    Start a new session with ``--files``, resume with ``--continue``
    or ``--resume``, and control state with ``--reset`` / ``--enquery``.
    """
    asyncio.run(
        _discuss_async(
            name,
            files,
            source,
            continue_session,
            resume,
            reset,
            enquery,
            output,
            stt,
            tts,
            verbose,
        )
    )


def _find_git_root() -> Path:
    """Find the git repository root from the current directory.

    Returns:
        Path to the git root, or current working directory if not in a repo.
    """
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return Path(result.stdout.strip())
    return Path.cwd()


async def _discuss_async(
    name: str | None,
    files: str | None,
    source: str | None,
    continue_session: bool,
    resume: str | None,
    reset: bool,
    enquery: bool,
    output: str | None,
    stt_name: str | None,
    tts_name: str | None,
    verbose: bool,
) -> None:
    """Async implementation of the discuss command.

    Resolves session (new, continue, or resume), applies state overrides,
    sets up providers, and runs the state machine.

    Args:
        name: Optional session name.
        files: Research folder path.
        source: Source code root path.
        continue_session: Whether to continue the latest session.
        resume: Name of a specific session to resume.
        reset: Whether to reset state to prepare.
        enquery: Whether to jump to enquiry state.
        output: Optional output folder path.
        stt_name: Optional STT provider name override.
        tts_name: Optional TTS provider name override.
        verbose: Whether to show verbose output.
    """
    from neev_voice.discuss.names import generate_session_name
    from neev_voice.discuss.runner import DiscussRunner
    from neev_voice.discuss.session import SessionManager
    from neev_voice.discuss.state import DiscussState, StateStack

    configure_logging(json_logs=not verbose)
    settings = _get_settings()
    logger.info("discuss_command_started")

    session_mgr = SessionManager(base_dir=Path(settings.discuss_base_dir))

    # Resolve session
    if resume:
        session = session_mgr.load_session(resume)
        if not session:
            console.print(f"[red]Error:[/red] Session '{resume}' not found.")
            existing = session_mgr.list_sessions()
            if existing:
                console.print(f"[dim]Available sessions: {', '.join(existing)}[/dim]")
            raise typer.Exit(1)
    elif continue_session:
        session = session_mgr.get_latest_session()
        if not session:
            console.print("[red]Error:[/red] No previous session found.")
            raise typer.Exit(1)
        console.print(f"[dim]Continuing session: {session.name}[/dim]")
    else:
        # New session
        if not files:
            console.print("[red]Error:[/red] --files is required for new sessions.")
            raise typer.Exit(1)

        files_path = Path(files)
        if not files_path.exists():
            console.print(f"[red]Error:[/red] Research path not found: {files}")
            raise typer.Exit(1)

        session_name = name or generate_session_name()
        source_path = source or str(_find_git_root())
        output_path = output

        try:
            session = session_mgr.create_session(
                session_name,
                research_path=str(files_path.resolve()),
                source_path=source_path,
                output_path=output_path,
            )
        except FileExistsError:
            console.print(
                f"[red]Error:[/red] Session '{session_name}' already exists. "
                "Use --resume to resume it."
            )
            raise typer.Exit(1) from None

        console.print(f"[bold]Created session:[/bold] {session_name}")

    # Apply state overrides
    if reset:
        session.state = DiscussState.PREPARE
        session.state_stack = StateStack()
        session_mgr.save_session(session)
        console.print("[dim]Session reset to prepare state.[/dim]")
    elif enquery:
        session.state = DiscussState.ENQUIRY
        session_mgr.save_session(session)
        console.print("[dim]Session set to enquiry state.[/dim]")

    # Setup STT/TTS providers
    stt_provider = None
    tts_provider = None

    stt_provider_name = stt_name or settings.stt_provider.value
    tts_provider_name = tts_name or settings.tts_provider.value

    try:
        from neev_voice.stt.sarvam import get_stt_provider
        from neev_voice.tts.edge import get_tts_provider

        stt_provider = get_stt_provider(stt_provider_name, settings)
        tts_provider = get_tts_provider(tts_provider_name, settings)
    except (ValueError, NeevConfigError) as e:
        if verbose:
            console.print(f"[yellow]Warning:[/yellow] Provider setup: {e}")
        # Continue without providers — prepare state doesn't need them

    console.print(
        f"[bold cyan]Session:[/bold cyan] {session.name} | "
        f"[bold cyan]State:[/bold cyan] {session.state}"
    )

    # Run the state machine
    runner = DiscussRunner(
        session=session,
        settings=settings,
        session_manager=session_mgr,
        tts_provider=tts_provider,
        stt_provider=stt_provider,
    )

    try:
        await runner.run()
    except (NeevError, RuntimeError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None

    console.print(f"\n[bold]Session complete:[/bold] {session.name}")
    if verbose:
        console.print(f"[dim]Session dir: {session_mgr.session_dir(session.name)}[/dim]")


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
    table.add_row("Enrichment Version", settings.enrichment_version.value)
    table.add_row("Enrichment Max Iterations", str(settings.enrichment_max_iterations))
    table.add_row("Discuss Base Dir", settings.discuss_base_dir)
    table.add_row(
        "Discuss MCP Config",
        settings.discuss_mcp_config or "(~/.config/mcphub/servers.json)",
    )
    table.add_row("Discuss Max Doc Chars", f"{settings.discuss_max_doc_chars:,}")
    table.add_row("Discuss Max Source Chars", f"{settings.discuss_max_source_chars:,}")
    table.add_row("Discuss Doc Extensions", settings.discuss_doc_extensions)
    table.add_row(
        "Discuss Prepare Model",
        settings.discuss_prepare_model or f"(← {settings.claude_model})",
    )
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
        raise typer.Exit(1) from None
    except (ValueError, NeevConfigError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


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
        raise typer.Exit(1) from None


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

    for tp in TTSProviderType:
        if tp == TTSProviderType.SARVAM:
            notes = "Requires NEEV_SARVAM_API_KEY"
        elif tp == TTSProviderType.EDGE:
            notes = "Free, no API key needed"
        else:
            notes = ""
        table.add_row("TTS", tp.value, notes)

    console.print(table)


if __name__ == "__main__":
    app()
