"""Transcript review gate module.

Provides an interactive review step between transcription and enrichment.
Users can accept, edit (via $EDITOR), or reject the transcript before
it proceeds to the enrichment agent.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from enum import StrEnum
from pathlib import Path

import structlog
from rich.console import Console
from rich.panel import Panel

from neev_voice.exceptions import TranscriptRejectedError

logger = structlog.get_logger(__name__)

__all__ = ["TranscriptReviewAction", "TranscriptReviewer"]


class TranscriptReviewAction(StrEnum):
    """Actions available during transcript review."""

    ACCEPT = "accept"
    EDIT = "edit"
    REJECT = "reject"


class TranscriptReviewer:
    """Interactive transcript reviewer.

    Displays the transcript and prompts the user to accept, edit, or reject
    it before proceeding to the enrichment layer.

    Args:
        console: Rich Console instance for display. Uses a new one if not provided.
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    async def review(
        self, transcript: str, transcript_path: Path
    ) -> tuple[TranscriptReviewAction, str]:
        """Present the transcript for user review.

        Displays the transcript and prompts the user to accept, edit in
        $EDITOR, or reject it.

        Args:
            transcript: The transcribed text to review.
            transcript_path: Path to the transcript file in .scratch/.

        Returns:
            Tuple of (action taken, final transcript text).

        Raises:
            TranscriptRejectedError: If the user rejects the transcript.
        """
        self._display_transcript(transcript)
        action = self._prompt_action()

        if action == TranscriptReviewAction.ACCEPT:
            logger.info("transcript_review", action="accept")
            return action, transcript

        if action == TranscriptReviewAction.EDIT:
            logger.info("transcript_review", action="edit", path=str(transcript_path))
            edited = await self._open_editor(transcript_path)
            return action, edited

        # REJECT
        logger.info("transcript_review", action="reject")
        raise TranscriptRejectedError("Transcript rejected by user.")

    def _display_transcript(self, text: str) -> None:
        """Display the transcript in a Rich panel.

        Args:
            text: The transcript text to display.
        """
        self._console.print()
        self._console.print(
            Panel(text, title="Transcription", border_style="yellow", padding=(1, 2))
        )

    def _prompt_action(self) -> TranscriptReviewAction:
        """Prompt the user to accept, edit, or reject the transcript.

        Loops until a valid choice is entered.

        Returns:
            The chosen TranscriptReviewAction.
        """
        valid = {
            "a": TranscriptReviewAction.ACCEPT,
            "e": TranscriptReviewAction.EDIT,
            "r": TranscriptReviewAction.REJECT,
        }
        while True:
            self._console.print(
                "[bold][A][/bold]ccept  |  [bold][E][/bold]dit  |  [bold][R][/bold]eject",
                style="dim",
            )
            choice = input("> ").strip().lower()
            if choice in valid:
                return valid[choice]
            self._console.print("[red]Invalid choice. Enter A, E, or R.[/red]")

    async def _open_editor(self, path: Path) -> str:
        """Open the transcript file in the user's editor.

        Resolves the editor from $VISUAL, $EDITOR, or falls back to ``vi``.
        Blocks until the editor process exits, then reads the file back.

        Args:
            path: Path to the transcript file to edit.

        Returns:
            The edited transcript text.
        """
        editor = self._resolve_editor()
        logger.info("opening_editor", editor=editor, path=str(path))
        await asyncio.to_thread(subprocess.run, [editor, str(path)], check=True)
        return path.read_text().strip()

    @staticmethod
    def _resolve_editor() -> str:
        """Resolve the user's preferred text editor.

        Checks $VISUAL, then $EDITOR, then falls back to ``vi``.

        Returns:
            Editor command name.
        """
        return os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
