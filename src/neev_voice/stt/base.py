"""Abstract base class for Speech-to-Text providers.

Defines the interface that all STT providers must implement,
along with the TranscriptionResult data model.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

__all__ = ["STTProvider", "TranscriptionResult"]


@dataclass
class TranscriptionResult:
    """Result from a speech-to-text transcription.

    Attributes:
        text: The transcribed text.
        language: Detected or configured language code.
        confidence: Confidence score from the provider (0.0 to 1.0).
        provider: Name of the STT provider that produced this result.
    """

    text: str
    language: str
    confidence: float
    provider: str


class STTProvider(ABC):
    """Abstract base class for Speech-to-Text providers.

    All STT providers must implement the transcribe method to convert
    audio files to text.
    """

    @abstractmethod
    async def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to the audio file to transcribe.

        Returns:
            TranscriptionResult containing the transcribed text and metadata.

        Raises:
            FileNotFoundError: If the audio file does not exist.
            RuntimeError: If transcription fails.
        """
        ...
