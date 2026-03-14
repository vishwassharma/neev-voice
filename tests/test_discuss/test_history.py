"""Tests for session conversation history."""

from __future__ import annotations

import json
from pathlib import Path

from neev_voice.discuss.history import SessionHistory


class TestSessionHistory:
    """Tests for SessionHistory."""

    def test_load_empty(self, tmp_path: Path) -> None:
        """Load returns empty list when no history exists."""
        history = SessionHistory(tmp_path)
        assert history.load() == []

    def test_append_and_load(self, tmp_path: Path) -> None:
        """Append adds entry that can be loaded."""
        history = SessionHistory(tmp_path)
        history.append("question", "What is X?")

        entries = history.load()
        assert len(entries) == 1
        assert entries[0]["type"] == "question"
        assert entries[0]["content"] == "What is X?"
        assert "timestamp" in entries[0]

    def test_append_multiple(self, tmp_path: Path) -> None:
        """Multiple appends accumulate entries."""
        history = SessionHistory(tmp_path)
        history.append("question", "Q1")
        history.append("answer", "A1")
        history.append("question", "Q2")

        entries = history.load()
        assert len(entries) == 3
        assert entries[0]["type"] == "question"
        assert entries[1]["type"] == "answer"
        assert entries[2]["type"] == "question"

    def test_persists_to_disk(self, tmp_path: Path) -> None:
        """History is persisted to history.json."""
        history = SessionHistory(tmp_path)
        history.append("question", "Test")

        data = json.loads((tmp_path / "history.json").read_text())
        assert len(data) == 1
        assert data[0]["content"] == "Test"

    def test_load_corrupt_file(self, tmp_path: Path) -> None:
        """Load returns empty list for corrupt file."""
        (tmp_path / "history.json").write_text("not json")
        history = SessionHistory(tmp_path)
        assert history.load() == []

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Append creates parent directories if needed."""
        session_dir = tmp_path / "nested" / "session"
        history = SessionHistory(session_dir)
        history.append("concept", "Concept A")

        assert (session_dir / "history.json").exists()

    def test_entry_has_timestamp(self, tmp_path: Path) -> None:
        """Each entry has an ISO timestamp."""
        history = SessionHistory(tmp_path)
        history.append("answer", "The answer")

        entries = history.load()
        assert "T" in entries[0]["timestamp"]  # ISO format
