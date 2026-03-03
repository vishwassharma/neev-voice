"""Speech-to-text provider module."""

from neev_voice.stt.base import STTProvider, TranscriptionResult
from neev_voice.stt.sarvam import SarvamSTT, get_stt_provider

__all__ = ["STTProvider", "SarvamSTT", "TranscriptionResult", "get_stt_provider"]
