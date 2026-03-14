"""Rich TUI panels for the discuss state machine.

Provides panel-building functions for consistent visual feedback
across presentation, enquiry, and answer playback states.
"""

from __future__ import annotations

from rich.panel import Panel
from rich.text import Text

from neev_voice.audio.keyboard import RecordingState

__all__ = [
    "make_answer_panel",
    "make_enquiry_panel",
    "make_presentation_panel",
    "make_recording_panel",
]


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
        lines.append("cancel", style="dim")
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
        lines.append("cancel", style="dim")
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
