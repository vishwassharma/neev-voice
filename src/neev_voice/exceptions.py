"""Custom exception hierarchy for neev-voice.

All application-specific exceptions inherit from :class:`NeevError`
so callers can catch the entire family with a single ``except`` clause
while still handling specific sub-categories as needed.
"""

__all__ = [
    "NeevConfigError",
    "NeevError",
    "NeevLLMError",
    "NeevSTTError",
    "NeevTTSError",
    "RecordingCancelledError",
    "TranscriptRejectedError",
]


class NeevError(Exception):
    """Base exception for all neev-voice errors."""


class NeevConfigError(NeevError):
    """Raised for configuration-related errors.

    Examples: missing API keys, invalid settings values,
    unreadable config files.
    """


class NeevSTTError(NeevError):
    """Raised for Speech-to-Text errors.

    Examples: API failures, transcription errors, audio format issues.
    """


class NeevTTSError(NeevError):
    """Raised for Text-to-Speech errors.

    Examples: API failures, synthesis errors, missing audio data.
    """


class NeevLLMError(NeevError):
    """Raised for LLM / enrichment agent errors.

    Examples: API failures, response parsing errors, missing API keys.
    """


class RecordingCancelledError(NeevError):
    """Raised when the user cancels a push-to-talk recording with ESC."""


class TranscriptRejectedError(NeevError):
    """Raised when the user rejects a transcript during review."""
