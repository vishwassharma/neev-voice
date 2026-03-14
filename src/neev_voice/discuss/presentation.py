"""Presentation engine for the discuss state machine.

Handles TTS synthesis and playback of concepts with interruptible
playback support. Waits for ENTER before each concept's TTS playback.
Monitors keyboard for spacebar (interrupt to enquiry),
ENTER (start/next concept), and ESC (cancel).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from neev_voice.discuss.session import SessionInfo

if TYPE_CHECKING:
    from neev_voice.config import NeevSettings
    from neev_voice.tts.base import TTSProvider

logger = structlog.get_logger(__name__)

__all__ = ["PresentationEngine", "PresentationResult"]


@dataclass
class PresentationResult:
    """Result of a presentation run.

    Attributes:
        interrupted: True if the user interrupted with spacebar.
        completed: True if all content was presented.
        cancelled: True if the user pressed ESC.
        state_data: State data for stack push (concept index, position, etc.).
    """

    interrupted: bool = False
    completed: bool = False
    cancelled: bool = False
    state_data: dict[str, Any] = field(default_factory=dict)


class PresentationEngine:
    """Presents concepts via TTS with interruptible playback.

    Loads TTS-ready transcripts and presents them one concept at a time.
    Waits for ENTER before starting TTS for each concept.
    Supports interruption (spacebar → enquiry) and cancellation (ESC → exit).

    Attributes:
        session: Current discuss session info.
        settings: Application settings.
        tts_provider: Text-to-speech provider for audio synthesis.
        prepare_dir: Path to the prepare phase output directory.
        on_concept_done: Optional callback fired after each concept completes.
    """

    def __init__(
        self,
        session: SessionInfo,
        settings: NeevSettings,
        tts_provider: TTSProvider | None = None,
        prepare_dir: Path | None = None,
        on_concept_done: Callable[[int], None] | None = None,
    ) -> None:
        """Initialize the presentation engine.

        Args:
            session: Current discuss session info.
            settings: Application settings.
            tts_provider: TTS provider for audio synthesis.
            prepare_dir: Override prepare directory path.
            on_concept_done: Optional callback called with concept index
                after each concept's playback completes successfully.
        """
        self.session = session
        self.settings = settings
        self.tts_provider = tts_provider
        self.prepare_dir = prepare_dir or Path(settings.discuss_base_dir) / session.name / "prepare"
        self.on_concept_done = on_concept_done

    def load_transcript(self, concept_index: int) -> str | None:
        """Load the TTS-ready transcript for a concept.

        Args:
            concept_index: Zero-based concept index.

        Returns:
            Transcript text, or None if not found.
        """
        transcripts_dir = self.prepare_dir / "transcripts"
        if not transcripts_dir.exists():
            return None

        # Find the transcript file matching this index
        prefix = f"{concept_index:03d}_"
        for f in sorted(transcripts_dir.iterdir()):
            if f.name.startswith(prefix) and f.is_file():
                return f.read_text(encoding="utf-8").strip()
        return None

    def list_concepts(self) -> list[dict[str, Any]]:
        """Load the concept list from concepts.json.

        Returns:
            List of concept dictionaries, or empty list if not found.
        """
        import json

        concepts_file = self.prepare_dir / "concepts.json"
        if not concepts_file.exists():
            return []
        try:
            data = json.loads(concepts_file.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, TypeError):
            return []

    async def run(self, start_index: int = 0) -> PresentationResult:
        """Present concepts sequentially with interruptible TTS playback.

        For each concept starting from start_index:
        1. Wait for ENTER before starting TTS
        2. Load the TTS-ready transcript
        3. Synthesize audio via TTS provider
        4. Play audio with keyboard monitoring
        5. On SPACEBAR (during wait or playback): interrupt → enquiry
        6. On ESC: cancel

        Args:
            start_index: Zero-based index of the first concept to present.

        Returns:
            PresentationResult indicating how the presentation ended.
        """
        concepts = self.list_concepts()
        if not concepts:
            logger.warning("presentation_no_concepts")
            return PresentationResult(completed=True)

        total = len(concepts)
        logger.info(
            "presentation_started",
            start=start_index,
            total=total,
        )

        for idx in range(start_index, total):
            concept = concepts[idx]
            transcript = self.load_transcript(idx)

            if not transcript:
                logger.warning("presentation_missing_transcript", index=idx)
                continue

            logger.info(
                "presentation_concept",
                index=idx,
                title=concept.get("title", f"Concept {idx}"),
            )

            # Wait for user to press ENTER before starting TTS
            gate_result = await self._wait_for_start(idx, total)
            if gate_result is not None:
                return gate_result

            result = await self._present_single(transcript, idx, total)

            if result.interrupted:
                result.state_data = {
                    "current_concept_index": idx,
                    "total_concepts": total,
                }
                return result

            if result.cancelled:
                return result

            # Playback complete — notify and continue to next concept
            if self.on_concept_done:
                self.on_concept_done(idx)

        return PresentationResult(completed=True)

    async def run_answer(self, answer_text: str) -> PresentationResult:
        """Present an enquiry answer via TTS.

        Similar to run() but presents a single piece of text (the
        answer to a user's enquiry) rather than concept transcripts.
        Waits for ENTER before starting TTS playback.

        Args:
            answer_text: The answer text to present.

        Returns:
            PresentationResult indicating how the presentation ended.
        """
        if not answer_text:
            return PresentationResult(completed=True)

        # Wait for user to press ENTER before speaking the answer
        gate_result = await self._wait_for_start(0, 1)
        if gate_result is not None:
            return gate_result

        return await self._present_single(answer_text, 0, 1)

    async def _wait_for_start(self, concept_index: int, total: int) -> PresentationResult | None:
        """Wait for user to press ENTER before starting TTS playback.

        Monitors keyboard for ENTER (proceed), SPACE (interrupt to enquiry),
        or ESC (cancel). Blocks until one of these keys is pressed.

        Args:
            concept_index: Current concept index (for state_data on interrupt).
            total: Total number of concepts.

        Returns:
            None if ENTER pressed (proceed to playback).
            PresentationResult if SPACE (interrupted) or ESC (cancelled).
        """
        import asyncio

        from rich.console import Console

        from neev_voice.audio.keyboard import KeyboardMonitor, MonitorMode

        console = Console()
        console.print(
            f"\n[dim]Concept {concept_index + 1}/{total}[/dim]  "
            "[bold green]ENTER[/bold green] to start  "
            "[bold yellow]SPACE[/bold yellow] to ask  "
            "[bold red]ESC[/bold red] to quit",
        )

        logger.info(
            "presentation_waiting_for_start",
            index=concept_index,
            total=total,
        )

        monitor = KeyboardMonitor(mode=MonitorMode.PRESENTATION)
        monitor.start()

        try:
            while True:
                if monitor.done_event.is_set():
                    return None
                if monitor.interrupted_event.is_set():
                    return PresentationResult(
                        interrupted=True,
                        state_data={
                            "current_concept_index": concept_index,
                            "total_concepts": total,
                        },
                    )
                if monitor.cancelled_event.is_set():
                    return PresentationResult(cancelled=True)
                await asyncio.sleep(0.05)
        finally:
            monitor.stop()

    async def _present_single(self, text: str, index: int, total: int) -> PresentationResult:
        """Present a single piece of text via TTS with keyboard monitoring.

        Args:
            text: Text to synthesize and play.
            index: Current item index (for state saving).
            total: Total number of items.

        Returns:
            PresentationResult for this segment.
        """
        # Synthesize TTS
        audio_path: Path | None = None
        if self.tts_provider:
            try:
                audio_path = await self.tts_provider.synthesize(text)
            except Exception as e:
                logger.error("presentation_tts_error", error=str(e))
                return PresentationResult(completed=True)

        # Play audio with keyboard monitoring
        if audio_path:
            return await self._play_interruptible(audio_path, index, total)

        # No TTS — just log and continue
        logger.info("presentation_no_audio", index=index)
        return PresentationResult(completed=False)

    async def _play_interruptible(
        self, audio_path: Path, index: int, total: int
    ) -> PresentationResult:
        """Play audio with interruptible keyboard monitoring.

        Runs audio playback in background and monitors keyboard for
        SPACEBAR (interrupt), ENTER (skip), or ESC (cancel).

        Args:
            audio_path: Path to the audio file to play.
            index: Current concept index.
            total: Total number of concepts.

        Returns:
            PresentationResult based on user interaction.
        """
        import sounddevice as sd
        from scipy.io import wavfile

        from neev_voice.audio.keyboard import KeyboardMonitor, MonitorMode

        try:
            sample_rate, data = wavfile.read(str(audio_path))
        except Exception as e:
            logger.error("presentation_audio_read_error", error=str(e))
            return PresentationResult(completed=False)

        # Convert to float32 if needed
        import numpy as np

        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0

        # Start playback in background
        sd.play(data, sample_rate)

        # Monitor keyboard while playing
        monitor = KeyboardMonitor(mode=MonitorMode.PRESENTATION)
        monitor.start()

        try:
            while sd.get_stream().active:
                if monitor.interrupted_event.wait(timeout=0.05):
                    sd.stop()
                    return PresentationResult(
                        interrupted=True,
                        state_data={
                            "current_concept_index": index,
                            "total_concepts": total,
                        },
                    )
                if monitor.done_event.wait(timeout=0.0):
                    sd.stop()
                    return PresentationResult(completed=False)  # skip to next
                if monitor.cancelled_event.wait(timeout=0.0):
                    sd.stop()
                    return PresentationResult(cancelled=True)
        finally:
            monitor.stop()

        # Playback completed naturally
        return PresentationResult(completed=False)  # move to next concept
