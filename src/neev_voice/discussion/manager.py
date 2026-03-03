"""Document discussion orchestrator module.

Manages the flow of interactive document discussions: loading documents,
chunking into sections, reading them via TTS, capturing user voice
responses via push-to-talk, and classifying intent.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from neev_voice.audio.keyboard import RecordingState
from neev_voice.audio.recorder import AudioRecorder
from neev_voice.config import NeevSettings
from neev_voice.intent.extractor import IntentCategory, IntentExtractor
from neev_voice.stt.base import STTProvider
from neev_voice.tts.base import TTSProvider

__all__ = ["DiscussionManager", "DiscussionResult"]


@dataclass
class DiscussionResult:
    """Result from discussing a single section.

    Attributes:
        section: The section text that was discussed.
        user_response: The user's transcribed response.
        intent: The classified intent category.
        summary: Summary of the user's response.
    """

    section: str
    user_response: str
    intent: IntentCategory
    summary: str


class DiscussionManager:
    """Orchestrates interactive document discussions.

    Loads a document, chunks it into sections, reads each section
    via TTS, captures the user's voice response, and classifies
    their intent. Results are returned for the caller to persist
    via the scratch pad.

    Attributes:
        settings: Application settings.
        recorder: Audio recorder for capturing voice input.
        stt: Speech-to-text provider.
        tts: Text-to-speech provider.
        intent_extractor: Intent classification engine.
        _document_text: The loaded document text.
        _document_path: Path to the loaded document.
        _sections: Parsed document sections.
    """

    def __init__(
        self,
        settings: NeevSettings,
        recorder: AudioRecorder,
        stt: STTProvider,
        tts: TTSProvider,
        intent_extractor: IntentExtractor,
    ) -> None:
        """Initialize DiscussionManager with all required components.

        Args:
            settings: Application settings.
            recorder: Audio recorder for voice capture.
            stt: Speech-to-text provider.
            tts: Text-to-speech provider.
            intent_extractor: Intent extraction engine.
        """
        self.settings = settings
        self.recorder = recorder
        self.stt = stt
        self.tts = tts
        self.intent_extractor = intent_extractor
        self._document_text: str = ""
        self._document_path: str = ""
        self._sections: list[str] = []

    def load_document(self, path: Path) -> list[str]:
        """Load a document and split it into sections.

        Reads the document file and splits it into sections based on
        markdown headers or double newlines.

        Args:
            path: Path to the document file.

        Returns:
            List of section strings.

        Raises:
            FileNotFoundError: If the document file does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        self._document_text = path.read_text(encoding="utf-8")
        self._document_path = str(path)
        self._sections = self._chunk_document(self._document_text)
        return self._sections

    @staticmethod
    def _chunk_document(text: str) -> list[str]:
        """Split document text into logical sections.

        Splits on markdown headers (## or #) or double newlines,
        filtering out empty sections.

        Args:
            text: The full document text.

        Returns:
            List of non-empty section strings.
        """
        # Split on markdown headers
        sections = re.split(r"\n(?=#{1,3}\s)", text)

        # If no headers found, split on double newlines
        if len(sections) <= 1:
            sections = text.split("\n\n")

        return [s.strip() for s in sections if s.strip()]

    async def discuss_section(
        self,
        section: str,
        on_recording_state_change: Callable[[RecordingState], None] | None = None,
    ) -> DiscussionResult:
        """Discuss a single document section with the user.

        Reads the section via TTS, records the user's voice response
        using push-to-talk, transcribes it, and classifies the intent.

        Args:
            section: The section text to discuss.
            on_recording_state_change: Optional callback fired on
                recording state transitions for UI updates.

        Returns:
            DiscussionResult with the user's response and classification.
        """
        # Synthesize and play the section
        audio_path = await self.tts.synthesize(section)
        try:
            TTSProvider.play_audio(audio_path)
        finally:
            audio_path.unlink(missing_ok=True)

        # Record user response with push-to-talk
        segment = await self.recorder.record_push_to_talk(
            on_state_change=on_recording_state_change,
        )
        wav_path = AudioRecorder.save_wav(segment)

        try:
            # Transcribe
            transcription = await self.stt.transcribe(wav_path)
        finally:
            wav_path.unlink(missing_ok=True)

        # Extract intent in discussion context
        intent = await self.intent_extractor.extract_discussion_intent(transcription.text, section)

        return DiscussionResult(
            section=section,
            user_response=transcription.text,
            intent=intent.category,
            summary=intent.summary,
        )

    async def run_discussion(
        self,
        path: Path,
        on_recording_state_change: Callable[[RecordingState], None] | None = None,
    ) -> list[DiscussionResult]:
        """Run a full document discussion.

        Loads the document, iterates through each section, discusses
        it with the user via push-to-talk, and collects all results.

        Args:
            path: Path to the document to discuss.
            on_recording_state_change: Optional callback fired on
                recording state transitions for UI updates.

        Returns:
            List of DiscussionResult for each section discussed.
        """
        sections = self.load_document(path)
        results = []

        for section in sections:
            result = await self.discuss_section(
                section, on_recording_state_change=on_recording_state_change
            )
            results.append(result)

        return results
