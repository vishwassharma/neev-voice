"""Conversation history for discuss sessions.

Tracks questions, answers, and concept presentations in a
``history.json`` file within each session directory.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

__all__ = ["SessionHistory"]


class SessionHistory:
    """Append-only conversation history for a discuss session.

    Stores entries in ``history.json`` within the session directory.
    Each entry records a timestamp, type, and content.

    Attributes:
        history_path: Path to the history.json file.
    """

    def __init__(self, session_dir: Path) -> None:
        """Initialize the history tracker.

        Args:
            session_dir: Path to the session directory.
        """
        self.history_path = session_dir / "history.json"

    def append(self, entry_type: str, content: str) -> None:
        """Append a new entry to the conversation history.

        Args:
            entry_type: Entry type (``question``, ``answer``, ``concept``).
            content: Text content of the entry.
        """
        entries = self.load()
        entries.append(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "type": entry_type,
                "content": content,
            }
        )
        self._save(entries)
        logger.debug("history_appended", type=entry_type, length=len(content))

    def load(self) -> list[dict[str, Any]]:
        """Load all history entries from disk.

        Returns:
            List of entry dicts, or empty list if no history exists.
        """
        if not self.history_path.exists():
            return []
        try:
            data = json.loads(self.history_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, entries: list[dict[str, Any]]) -> None:
        """Persist entries to disk.

        Args:
            entries: Full list of history entries.
        """
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(entries, indent=2, default=str) + "\n"
        self.history_path.write_text(content, encoding="utf-8")
