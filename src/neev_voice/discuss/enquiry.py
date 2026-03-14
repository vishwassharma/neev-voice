"""Enquiry engine for the discuss state machine.

Handles user enquiry capture via voice recording (push-to-talk + STT)
or manual text input (editor). Reuses existing AudioRecorder, STT
provider, and TranscriptReviewer for the voice path.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click
import structlog

from neev_voice.audio.recorder import AudioRecorder, RecordingCancelledError
from neev_voice.discuss.session import SessionInfo

if TYPE_CHECKING:
    from neev_voice.config import NeevSettings
    from neev_voice.stt.base import STTProvider

logger = structlog.get_logger(__name__)

__all__ = ["EnquiryEngine", "EnquiryResult"]


@dataclass
class EnquiryResult:
    """Result of an enquiry capture.

    Attributes:
        escaped: True if the user pressed ESC to go back.
        query: The captured query text, or None if escaped.
        audio_path: Path to saved audio (voice path only), or None.
        source: How the query was captured ('voice', 'manual', or 'escaped').
    """

    escaped: bool
    query: str | None = None
    audio_path: Path | None = None
    source: str = "escaped"


class EnquiryEngine:
    """Captures user enquiries via voice recording or manual text input.

    In the enquiry state, the user can:
    - Press SPACEBAR to record a voice query (push-to-talk → STT → editor review)
    - Press M to type a query manually in the editor
    - Press ESC to escape back to the previous state

    Attributes:
        session: The current discuss session.
        settings: Application settings.
        stt_provider: Speech-to-text provider for voice transcription.
        session_dir: Path to the session's scratch pad directory.
    """

    def __init__(
        self,
        session: SessionInfo,
        settings: NeevSettings,
        stt_provider: STTProvider | None = None,
        session_dir: Path | None = None,
    ) -> None:
        """Initialize the enquiry engine.

        Args:
            session: Current discuss session info.
            settings: Application settings with audio/STT config.
            stt_provider: Optional STT provider (required for voice path).
            session_dir: Override session directory path.
        """
        self.session = session
        self.settings = settings
        self.stt_provider = stt_provider
        self.session_dir = session_dir or Path(settings.discuss_base_dir) / session.name

    async def run(self) -> EnquiryResult:
        """Run the enquiry capture flow.

        Waits for user input (SPACEBAR for voice, M for manual, ESC to escape),
        then captures and processes the query.

        Returns:
            EnquiryResult with the captured query or escape indication.
        """
        from neev_voice.audio.keyboard import KeyboardMonitor, MonitorMode

        logger.info("enquiry_engine_started", session=self.session.name)

        monitor = KeyboardMonitor(
            mode=MonitorMode.PRESENTATION,
            stdin=None,
        )
        monitor.start()

        try:
            # Wait for any event
            while True:
                if monitor.interrupted_event.wait(timeout=0.1):
                    # SPACEBAR pressed — voice recording
                    monitor.stop()
                    return await self._handle_voice()
                if monitor.manual_event.wait(timeout=0.0):
                    # M pressed — manual text entry
                    monitor.stop()
                    return self._handle_manual()
                if monitor.cancelled_event.wait(timeout=0.0):
                    # ESC pressed — escape back
                    monitor.stop()
                    return EnquiryResult(escaped=True, source="escaped")
                if monitor.done_event.wait(timeout=0.0):
                    # ENTER pressed — treat as escape
                    monitor.stop()
                    return EnquiryResult(escaped=True, source="escaped")
        except Exception:
            monitor.stop()
            raise

    async def _handle_voice(self) -> EnquiryResult:
        """Handle voice recording enquiry path.

        Records audio via push-to-talk, transcribes via STT, and opens
        editor for transcript review. If the user quits the editor
        without saving, resets and returns to enquiry state.

        Returns:
            EnquiryResult with captured voice query.
        """
        if self.stt_provider is None:
            logger.warning("no_stt_provider_for_voice_enquiry")
            return self._handle_manual()

        recorder = AudioRecorder(settings=self.settings)

        try:
            segment = await recorder.record_push_to_talk()
        except RecordingCancelledError:
            logger.info("voice_enquiry_cancelled")
            return EnquiryResult(escaped=False, source="voice")

        wav_path = AudioRecorder.save_wav(segment)

        try:
            transcription = await self.stt_provider.transcribe(wav_path)
        except Exception as e:
            logger.error("voice_enquiry_stt_error", error=str(e))
            wav_path.unlink(missing_ok=True)
            return EnquiryResult(escaped=False, source="voice")

        # Save audio to enquiry directory
        enquiry_dir = self._make_enquiry_dir()
        audio_dest = enquiry_dir / "audio.wav"
        shutil.copy2(str(wav_path), str(audio_dest))
        wav_path.unlink(missing_ok=True)

        # Open editor for transcript review/correction
        edited = click.edit(text=transcription.text, extension=".txt")

        if edited is None or not edited.strip():
            # User quit editor without saving — cleanup and reset
            logger.info("voice_enquiry_editor_cancelled")
            self._cleanup_enquiry_dir(enquiry_dir)
            return EnquiryResult(escaped=False, source="voice")

        query = edited.strip()

        # Save query text
        (enquiry_dir / "query.txt").write_text(query, encoding="utf-8")
        logger.info("voice_enquiry_captured", query_length=len(query))

        return EnquiryResult(
            escaped=False,
            query=query,
            audio_path=audio_dest,
            source="voice",
        )

    def _handle_manual(self) -> EnquiryResult:
        """Handle manual text entry enquiry path.

        Opens the editor for the user to type their query. If the user
        quits without saving, resets and continues in enquiry state.

        Returns:
            EnquiryResult with captured manual query.
        """
        text = click.edit(text="", extension=".txt")

        if text is None or not text.strip():
            logger.info("manual_enquiry_editor_cancelled")
            return EnquiryResult(escaped=False, source="manual")

        query = text.strip()

        # Save to enquiry directory
        enquiry_dir = self._make_enquiry_dir()
        (enquiry_dir / "query.txt").write_text(query, encoding="utf-8")
        logger.info("manual_enquiry_captured", query_length=len(query))

        return EnquiryResult(
            escaped=False,
            query=query,
            source="manual",
        )

    def _make_enquiry_dir(self) -> Path:
        """Create a timestamped enquiry directory.

        Returns:
            Path to the created enquiry directory.
        """
        from datetime import UTC, datetime

        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        enquiry_dir = self.session_dir / "enquiries" / timestamp
        enquiry_dir.mkdir(parents=True, exist_ok=True)
        return enquiry_dir

    def _cleanup_enquiry_dir(self, enquiry_dir: Path) -> None:
        """Remove an enquiry directory on cancellation.

        Args:
            enquiry_dir: Path to the enquiry directory to remove.
        """
        shutil.rmtree(enquiry_dir, ignore_errors=True)
