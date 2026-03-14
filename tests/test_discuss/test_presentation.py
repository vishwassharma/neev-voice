"""Tests for the presentation engine."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from neev_voice.discuss.presentation import PresentationEngine, PresentationResult
from neev_voice.discuss.session import SessionInfo


@pytest.fixture
def session(tmp_path: Path) -> SessionInfo:
    """Create a test session."""
    return SessionInfo(
        name="test-session",
        research_path=str(tmp_path / "research"),
        source_path=str(tmp_path / "source"),
        output_path=str(tmp_path / "output"),
    )


@pytest.fixture
def settings() -> MagicMock:
    """Create mock settings."""
    s = MagicMock()
    s.discuss_base_dir = ".scratch/neev/discuss"
    return s


@pytest.fixture
def prepare_dir(tmp_path: Path) -> Path:
    """Create a prepare directory with concepts and transcripts."""
    prep = tmp_path / "prepare"
    (prep / "transcripts").mkdir(parents=True)

    concepts = [
        {"index": 0, "title": "Concept A", "description": "First concept"},
        {"index": 1, "title": "Concept B", "description": "Second concept"},
        {"index": 2, "title": "Concept C", "description": "Third concept"},
    ]
    (prep / "concepts.json").write_text(json.dumps(concepts))

    (prep / "transcripts" / "000_concept-a.md").write_text(
        "Welcome to concept A. This explains the basics."
    )
    (prep / "transcripts" / "001_concept-b.md").write_text("Now let's learn about concept B.")
    (prep / "transcripts" / "002_concept-c.md").write_text(
        "Finally, concept C brings everything together."
    )
    return prep


class TestPresentationResult:
    """Tests for PresentationResult dataclass."""

    def test_defaults(self) -> None:
        """Default result is not interrupted, not completed, not cancelled."""
        result = PresentationResult()
        assert not result.interrupted
        assert not result.completed
        assert not result.cancelled
        assert result.state_data == {}

    def test_interrupted_result(self) -> None:
        """Interrupted result has state data."""
        result = PresentationResult(
            interrupted=True,
            state_data={"current_concept_index": 1, "total_concepts": 3},
        )
        assert result.interrupted
        assert result.state_data["current_concept_index"] == 1

    def test_completed_result(self) -> None:
        """Completed result."""
        result = PresentationResult(completed=True)
        assert result.completed


class TestPresentationEngine:
    """Tests for PresentationEngine."""

    def test_init(self, session: SessionInfo, settings: MagicMock, prepare_dir: Path) -> None:
        """Engine initializes correctly."""
        engine = PresentationEngine(session, settings, prepare_dir=prepare_dir)
        assert engine.session is session

    def test_list_concepts(
        self, session: SessionInfo, settings: MagicMock, prepare_dir: Path
    ) -> None:
        """list_concepts returns concept list from concepts.json."""
        engine = PresentationEngine(session, settings, prepare_dir=prepare_dir)
        concepts = engine.list_concepts()
        assert len(concepts) == 3
        assert concepts[0]["title"] == "Concept A"

    def test_list_concepts_empty(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """list_concepts returns empty list when no concepts.json."""
        engine = PresentationEngine(session, settings, prepare_dir=tmp_path)
        assert engine.list_concepts() == []

    def test_list_concepts_corrupted(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """list_concepts returns empty list for corrupted file."""
        (tmp_path / "concepts.json").write_text("not json")
        engine = PresentationEngine(session, settings, prepare_dir=tmp_path)
        assert engine.list_concepts() == []

    def test_load_transcript(
        self, session: SessionInfo, settings: MagicMock, prepare_dir: Path
    ) -> None:
        """load_transcript loads the correct transcript file."""
        engine = PresentationEngine(session, settings, prepare_dir=prepare_dir)
        text = engine.load_transcript(0)
        assert text is not None
        assert "concept A" in text

    def test_load_transcript_missing(
        self, session: SessionInfo, settings: MagicMock, prepare_dir: Path
    ) -> None:
        """load_transcript returns None for missing index."""
        engine = PresentationEngine(session, settings, prepare_dir=prepare_dir)
        assert engine.load_transcript(99) is None

    def test_load_transcript_no_dir(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """load_transcript returns None when transcripts dir doesn't exist."""
        engine = PresentationEngine(session, settings, prepare_dir=tmp_path)
        assert engine.load_transcript(0) is None


class TestPresentationEngineRun:
    """Tests for run() and run_answer() methods."""

    async def test_run_no_concepts(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """run() returns completed when no concepts exist."""
        engine = PresentationEngine(session, settings, prepare_dir=tmp_path)
        result = await engine.run()
        assert result.completed

    async def test_run_no_tts_provider(
        self, session: SessionInfo, settings: MagicMock, prepare_dir: Path
    ) -> None:
        """run() handles missing TTS provider gracefully."""
        engine = PresentationEngine(session, settings, tts_provider=None, prepare_dir=prepare_dir)
        # Without TTS, _present_single returns completed=False (no audio)
        # but loop continues through all concepts
        result = await engine.run()
        # Should complete after iterating through all concepts
        assert result.completed

    async def test_run_tts_error(
        self, session: SessionInfo, settings: MagicMock, prepare_dir: Path
    ) -> None:
        """run() handles TTS synthesis errors."""
        tts = AsyncMock()
        tts.synthesize = AsyncMock(side_effect=RuntimeError("TTS failed"))

        engine = PresentationEngine(session, settings, tts_provider=tts, prepare_dir=prepare_dir)
        # TTS error returns completed=True for the segment, loop continues
        result = await engine.run()
        assert result.completed

    async def test_run_answer_empty(
        self, session: SessionInfo, settings: MagicMock, prepare_dir: Path
    ) -> None:
        """run_answer() returns completed for empty text."""
        engine = PresentationEngine(session, settings, prepare_dir=prepare_dir)
        result = await engine.run_answer("")
        assert result.completed

    async def test_run_with_start_index(
        self, session: SessionInfo, settings: MagicMock, prepare_dir: Path
    ) -> None:
        """run() respects start_index parameter."""
        engine = PresentationEngine(session, settings, tts_provider=None, prepare_dir=prepare_dir)
        # Start from index 2 (last concept)
        result = await engine.run(start_index=2)
        assert result.completed

    async def test_run_missing_transcript_skipped(
        self, session: SessionInfo, settings: MagicMock, prepare_dir: Path
    ) -> None:
        """run() skips concepts with missing transcripts."""
        # Delete one transcript
        for f in (prepare_dir / "transcripts").iterdir():
            if f.name.startswith("001_"):
                f.unlink()

        engine = PresentationEngine(session, settings, tts_provider=None, prepare_dir=prepare_dir)
        result = await engine.run()
        assert result.completed
