"""Rich TUI for the discuss state machine.

Provides panel-building functions and the DiscussTUI wrapper class
for consistent visual feedback across all discuss states.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from neev_voice.audio.keyboard import RecordingState
from neev_voice.discuss.state import DiscussState

if TYPE_CHECKING:
    from neev_voice.discuss.runner import DiscussRunner

__all__ = [
    "DiscussTUI",
    "make_answer_panel",
    "make_enquiry_panel",
    "make_prepare_panel",
    "make_presentation_panel",
    "make_recording_panel",
]


def make_prepare_panel() -> Panel:
    """Create a Rich Panel for the prepare state.

    Returns:
        Styled Rich Panel showing preparation status.
    """
    lines = Text()
    lines.append("  Analyzing documents and extracting concepts...\n\n", style="bold")
    lines.append("  This may take a few minutes", style="dim")

    return Panel(lines, title="Preparing", border_style="magenta")


def make_prepare_enquiry_panel() -> Panel:
    """Create a Rich Panel for the prepare-enquiry state.

    Returns:
        Styled Rich Panel showing research status.
    """
    lines = Text()
    lines.append("  Researching answer...\n\n", style="bold")
    lines.append("  This may take a moment", style="dim")

    return Panel(lines, title="Researching", border_style="magenta")


def make_presentation_panel(
    title: str,
    description: str = "",
    index: int = 0,
    total: int = 1,
    playing: bool = False,
) -> Panel:
    """Create a Rich Panel for the presentation state.

    Args:
        title: Concept title.
        description: Concept description.
        index: Zero-based concept index.
        total: Total number of concepts.
        playing: True if TTS is currently playing.

    Returns:
        Styled Rich Panel.
    """
    lines = Text()

    if playing:
        lines.append("  Playing  ", style="bold green")
        lines.append(title, style="bold")
        lines.append("\n\n")
        lines.append("  SPACE ", style="bold yellow")
        lines.append("ask  ", style="dim")
        lines.append("ENTER ", style="bold green")
        lines.append("skip  ", style="dim")
        lines.append("ESC ", style="bold red")
        lines.append("cancel\n", style="dim")
        lines.append("  1 ", style="bold cyan")
        lines.append("1x  ", style="dim")
        lines.append("2 ", style="bold cyan")
        lines.append("1.25x  ", style="dim")
        lines.append("3 ", style="bold cyan")
        lines.append("1.5x  ", style="dim")
        lines.append("4 ", style="bold cyan")
        lines.append("2x", style="dim")
    else:
        lines.append(f"  {title}\n", style="bold")
        if description:
            lines.append(f"  {description}\n", style="dim")
        lines.append("\n")
        lines.append("  ENTER ", style="bold green")
        lines.append("start  ", style="dim")
        lines.append("SPACE ", style="bold yellow")
        lines.append("ask  ", style="dim")
        lines.append("ESC ", style="bold red")
        lines.append("quit", style="dim")

    panel_title = f"Concept {index + 1}/{total}"
    border = "green" if playing else "cyan"
    return Panel(lines, title=panel_title, border_style=border)


def make_enquiry_panel() -> Panel:
    """Create a Rich Panel for the enquiry state.

    Returns:
        Styled Rich Panel with key instructions.
    """
    lines = Text()
    lines.append("  Ask a question about the material\n\n", style="bold")
    lines.append("  SPACE ", style="bold yellow")
    lines.append("voice  ", style="dim")
    lines.append("M ", style="bold cyan")
    lines.append("manual  ", style="dim")
    lines.append("ESC ", style="bold red")
    lines.append("back", style="dim")

    return Panel(lines, title="Enquiry", border_style="yellow")


def make_recording_panel(state: RecordingState) -> Panel:
    """Create a Rich Panel showing push-to-talk recording state.

    Args:
        state: Current recording state.

    Returns:
        Styled Rich Panel with state-appropriate styling.
    """
    styles = {
        RecordingState.IDLE: ("  Waiting... Hold SPACEBAR to record", "dim"),
        RecordingState.RECORDING: ("  RECORDING  Hold SPACEBAR...", "bold red"),
        RecordingState.PAUSED: (
            "  PAUSED  Hold SPACEBAR to continue, ENTER to send, ESC to cancel",
            "bold yellow",
        ),
        RecordingState.DONE: ("  Captured! Processing...", "bold green"),
        RecordingState.CANCELLED: ("  Cancelled.", "bold dim"),
    }
    text, style = styles.get(state, ("", ""))
    return Panel(Text(text, style=style), title="Voice Recording", border_style="cyan")


def make_answer_panel(playing: bool = False) -> Panel:
    """Create a Rich Panel for the presentation-enquiry (answer) state.

    Args:
        playing: True if answer TTS is currently playing.

    Returns:
        Styled Rich Panel.
    """
    lines = Text()

    if playing:
        lines.append("  Speaking answer...\n\n", style="bold green")
        lines.append("  SPACE ", style="bold yellow")
        lines.append("ask  ", style="dim")
        lines.append("ENTER ", style="bold green")
        lines.append("skip  ", style="dim")
        lines.append("ESC ", style="bold red")
        lines.append("cancel\n", style="dim")
        lines.append("  1 ", style="bold cyan")
        lines.append("1x  ", style="dim")
        lines.append("2 ", style="bold cyan")
        lines.append("1.25x  ", style="dim")
        lines.append("3 ", style="bold cyan")
        lines.append("1.5x  ", style="dim")
        lines.append("4 ", style="bold cyan")
        lines.append("2x", style="dim")
    else:
        lines.append("  Answer ready\n\n", style="bold")
        lines.append("  ENTER ", style="bold green")
        lines.append("listen  ", style="dim")
        lines.append("SPACE ", style="bold yellow")
        lines.append("ask  ", style="dim")
        lines.append("ESC ", style="bold red")
        lines.append("quit", style="dim")

    border = "green" if playing else "cyan"
    return Panel(lines, title="Answer", border_style=border)


_STATE_PANELS = {
    DiscussState.PREPARE: make_prepare_panel,
    DiscussState.ENQUIRY: make_enquiry_panel,
    DiscussState.PREPARE_ENQUIRY: make_prepare_enquiry_panel,
}
"""Static panel builders for states that don't need dynamic parameters."""

_MAX_ANSWER_DISPLAY = 2000
"""Maximum characters of answer text to display in the panel."""


def make_answer_text_panel(answer: str) -> Panel:
    """Create a Rich Panel displaying the answer text content.

    Args:
        answer: The answer text to display.

    Returns:
        Styled Rich Panel with answer content and key instructions.
    """
    display = answer[:_MAX_ANSWER_DISPLAY]
    if len(answer) > _MAX_ANSWER_DISPLAY:
        display += "\n..."

    lines = Text()
    lines.append(f"{display}\n\n", style="")
    lines.append("  SPACE ", style="bold yellow")
    lines.append("follow-up  ", style="dim")
    lines.append("ESC ", style="bold red")
    lines.append("done", style="dim")

    return Panel(lines, title="Answer", border_style="green")


class DiscussTUI:
    """Rich TUI wrapper for the discuss state machine.

    Wraps a DiscussRunner, observing state transitions via the
    ``on_state_enter`` callback and rendering state-specific panels.
    Engines remain logic-only; the TUI handles all visual feedback.

    Attributes:
        runner: The wrapped DiscussRunner instance.
        console: Rich Console for output.
    """

    def __init__(self, runner: DiscussRunner, console: Console | None = None) -> None:
        """Initialize the TUI wrapper.

        Args:
            runner: DiscussRunner to wrap.
            console: Optional Rich Console (created if not provided).
        """
        self.runner = runner
        self.console = console or Console()
        self.runner.on_state_enter = self._on_state_enter
        self._active_status: Any | None = None

    async def run(self) -> None:
        """Run the discuss session with TUI display.

        Shows a welcome banner, renders state panels during the
        session, and prints a goodbye message on exit.
        """
        self._print_header()
        try:
            await self.runner.run()
        finally:
            self._stop_spinner()
        self._print_footer()

    _SPINNER_STATES: ClassVar[dict[DiscussState, str]] = {
        DiscussState.PREPARE: "Analyzing documents and extracting concepts...",
        DiscussState.PREPARE_ENQUIRY: "Researching answer...",
    }
    """States that show a spinner during engine execution."""

    def _stop_spinner(self) -> None:
        """Stop any active spinner from a previous state."""
        if self._active_status is not None:
            self._active_status.stop()
            self._active_status = None

    def _on_state_enter(self, state: DiscussState, ctx: dict) -> None:
        """Callback fired at the start of each state machine iteration.

        Shows spinners for long-running states, answer text for
        PRESENTATION_ENQUIRY, and static panels for others.

        Args:
            state: The state being entered.
            ctx: Context dict with state-specific data (e.g. answer text).
        """
        self._stop_spinner()

        # Spinner states (prepare, prepare-enquiry)
        spinner_msg = self._SPINNER_STATES.get(state)
        if spinner_msg:
            if state == DiscussState.PREPARE_ENQUIRY:
                query = ctx.get("query", "")
                if query:
                    truncated = query[:80] + ("..." if len(query) > 80 else "")
                    self.console.print(f"\n[bold]Q:[/bold] [italic]{truncated}[/italic]")
            self._active_status = self.console.status(spinner_msg, spinner="dots")
            self._active_status.start()
            return

        if state == DiscussState.PRESENTATION_ENQUIRY:
            answer = ctx.get("answer", "")
            if answer:
                self.console.print(make_answer_text_panel(answer))
            return

        if state == DiscussState.PRESENTATION:
            # Presentation panel is rendered by the engine's _wait_for_start
            return

        if state == DiscussState.ENQUIRY:
            self.console.print(make_enquiry_panel())
            return

    def _print_header(self) -> None:
        """Print session header with name and initial state."""
        session = self.runner.session
        header = Text()
        header.append("  neev discuss", style="bold cyan")
        header.append(f"  {session.name}", style="bold")
        header.append(f"  ({session.state.value})", style="dim")
        self.console.print(Panel(header, border_style="cyan"))

    def _print_footer(self) -> None:
        """Print session complete message."""
        session = self.runner.session
        self.console.print(f"\n[bold]Session complete:[/bold] {session.name}")
