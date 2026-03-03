"""Text-to-speech provider module."""

from neev_voice.tts.base import TTSProvider
from neev_voice.tts.edge import EdgeTTS, get_tts_provider
from neev_voice.tts.sarvam import SarvamTTS

__all__ = ["EdgeTTS", "SarvamTTS", "TTSProvider", "get_tts_provider"]
