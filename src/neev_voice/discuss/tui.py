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
    "EQUALIZER_FRAMES",
    "DiscussTUI",
    "get_equalizer_frame",
    "make_answer_panel",
    "make_answer_text_panel",
    "make_enquiry_panel",
    "make_playback_panel",
    "make_prepare_panel",
    "make_presentation_panel",
    "make_recording_animated_panel",
    "make_recording_panel",
]

EQUALIZER_FRAMES = [
    "▁ ▃ ▅ ▇ ▅ ▃ ▁",
    "▃ ▅ ▇ ▅ ▃ ▁ ▃",
    "▅ ▇ ▅ ▃ ▁ ▃ ▅",
    "▇ ▅ ▃ ▁ ▃ ▅ ▇",
    "▅ ▃ ▁ ▃ ▅ ▇ ▅",
    "▃ ▁ ▃ ▅ ▇ ▅ ▃",
]
"""Fallback animation frames when no real audio level is available."""

_BARS = " ▁▂▃▄▅▆▇"
"""Unicode block characters ordered by height (9 levels)."""

SPEED_LABELS = {1.0: "1x", 1.25: "1.25x", 1.5: "1.5x", 2.0: "2x"}
"""Display labels for playback speeds."""

_NUM_EQ_BARS = 7
"""Number of bars in the equalizer display."""


def get_equalizer_frame(tick: int) -> str:
    """Get a preset equalizer animation frame for a given tick.

    Fallback when no real audio level data is available.

    Args:
        tick: Animation tick counter (increments each refresh).

    Returns:
        Unicode equalizer bar string for the current frame.
    """
    return EQUALIZER_FRAMES[tick % len(EQUALIZER_FRAMES)]


def level_to_bars(level: float, num_bars: int = _NUM_EQ_BARS) -> str:
    """Convert an audio RMS level (0.0-1.0) to unicode equalizer bars.

    Generates bars of varying height based on the level, with a
    peak in the center and taper at the edges for a natural look.

    Args:
        level: Audio RMS level clamped to 0.0-1.0.
        num_bars: Number of bars to generate.

    Returns:
        Space-separated unicode bar string.
    """
    import random

    level = max(0.0, min(1.0, level))
    max_idx = len(_BARS) - 1
    bars = []
    for i in range(num_bars):
        # Taper: center bars higher, edges lower
        distance = abs(i - num_bars // 2) / (num_bars // 2)
        taper = 1.0 - distance * 0.4
        height = level * taper
        # Add small random jitter for natural look
        jitter = random.uniform(-0.1, 0.1)  # noqa: S311
        idx = int(max(0.0, min(1.0, height + jitter)) * max_idx)
        bars.append(_BARS[idx])
    return " ".join(bars)


_SPEED_KEYS = [
    ("1", 1.0, "1x"),
    ("2", 1.25, "1.25x"),
    ("3", 1.5, "1.5x"),
    ("4", 2.0, "2x"),
]
"""Speed key mappings: (key, speed_value, label)."""


def _append_speed_keys(lines: Text, active_speed: float) -> None:
    """Append speed key instructions with the active speed highlighted.

    Args:
        lines: Rich Text object to append to.
        active_speed: Currently active playback speed.
    """
    lines.append("  ")
    for key, spd, label in _SPEED_KEYS:
        if abs(spd - active_speed) < 0.01:
            lines.append(f"{key} ", style="bold white on cyan")
            lines.append(f"{label}  ", style="bold cyan")
        else:
            lines.append(f"{key} ", style="dim")
            lines.append(f"{label}  ", style="dim")


def make_playback_panel(
    title: str = "Answer",
    speed: float = 1.0,
    tick: int = 0,
    index: int = 0,
    total: int = 1,
    is_answer: bool = False,
    level: float = -1.0,
) -> Panel:
    """Create an animated playback panel with equalizer and speed controls.

    Used during both concept presentation and answer playback. When
    ``level`` is provided (>= 0), the equalizer reflects actual audio
    amplitude. Otherwise falls back to preset animation frames.

    Args:
        title: Content title (concept name or "Answer").
        speed: Current playback speed.
        tick: Animation tick for equalizer fallback.
        index: Zero-based concept index.
        total: Total number of concepts.
        is_answer: True if playing an answer (vs concept).
        level: Audio RMS level (0.0-1.0). Negative = use preset frames.

    Returns:
        Styled Rich Panel with equalizer animation and key instructions.
    """
    eq = level_to_bars(level) if level >= 0 else get_equalizer_frame(tick)
    speed_label = SPEED_LABELS.get(speed, f"{speed}x")

    lines = Text()
    lines.append(f"  {eq}", style="bold green")
    lines.append(f"  {speed_label}\n", style="bold cyan")
    lines.append(f"  {title}\n\n", style="bold")
    lines.append("  SPACE ", style="bold yellow")
    lines.append("ask  ", style="dim")
    lines.append("ENTER ", style="bold green")
    lines.append("skip  ", style="dim")
    lines.append("ESC ", style="bold red")
    lines.append("cancel\n", style="dim")
    _append_speed_keys(lines, speed)

    panel_title = "Answer" if is_answer else f"Playing {index + 1}/{total}"
    return Panel(lines, title=panel_title, border_style="green")


def make_recording_animated_panel(
    state: RecordingState, tick: int = 0, level: float = -1.0
) -> Panel:
    """Create an animated recording panel with equalizer.

    Shows equalizer reflecting actual mic levels when ``level`` is
    provided, falls back to preset animation otherwise.

    Args:
        state: Current recording state.
        tick: Animation tick for equalizer fallback.
        level: Mic RMS level (0.0-1.0). Negative = use preset frames.

    Returns:
        Styled Rich Panel with recording state and animation.
    """
    lines = Text()

    if state == RecordingState.RECORDING:
        eq = level_to_bars(level) if level >= 0 else get_equalizer_frame(tick)
        lines.append(f"  {eq}", style="bold red")
        lines.append("  RECORDING\n\n", style="bold red")
        lines.append("  Release to pause, ", style="dim")
        lines.append("ENTER ", style="bold green")
        lines.append("send, ", style="dim")
        lines.append("ESC ", style="bold red")
        lines.append("cancel", style="dim")
    elif state == RecordingState.PAUSED:
        lines.append("  PAUSED\n\n", style="bold yellow")
        lines.append("  Hold ", style="dim")
        lines.append("SPACEBAR ", style="bold yellow")
        lines.append("to continue, ", style="dim")
        lines.append("ENTER ", style="bold green")
        lines.append("send, ", style="dim")
        lines.append("ESC ", style="bold red")
        lines.append("cancel", style="dim")
    elif state == RecordingState.DONE:
        lines.append("  Captured! Processing...", style="bold green")
    elif state == RecordingState.CANCELLED:
        lines.append("  Cancelled.", style="bold dim")
    else:
        lines.append("  Hold ", style="dim")
        lines.append("SPACEBAR ", style="bold yellow")
        lines.append("to record", style="dim")

    border = "red" if state == RecordingState.RECORDING else "cyan"
    return Panel(lines, title="Voice Recording", border_style=border)


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
