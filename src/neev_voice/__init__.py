"""Neev Voice - A Python CLI voice agent for Hindi-English mixed speech.

Listens to user voice, transcribes it, extracts intent, discusses plan
documents, detects agreement/disagreement, and saves results to scratch pad.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("neev-voice")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"
