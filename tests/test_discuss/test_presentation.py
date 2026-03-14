"""Tests for the presentation engine."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
        assert engine.on_concept_done is None

    def test_init_with_callback(
        self, session: SessionInfo, settings: MagicMock, prepare_dir: Path
    ) -> None:
        """Engine accepts on_concept_done callback."""
        cb = MagicMock()
        engine = PresentationEngine(session, settings, prepare_dir=prepare_dir, on_concept_done=cb)
        assert engine.on_concept_done is cb

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


class TestWaitForStart:
    """Tests for _wait_for_start() ENTER gate method."""

    @patch("neev_voice.audio.keyboard.KeyboardMonitor")
    @patch("rich.console.Console")
    async def test_enter_pressed_returns_none(
        self,
        mock_console_cls: MagicMock,
        mock_monitor_cls: MagicMock,
        session: SessionInfo,
        settings: MagicMock,
        prepare_dir: Path,
    ) -> None:
        """ENTER press returns None (proceed to playback)."""
        mock_monitor = MagicMock()
        mock_monitor.done_event.is_set.return_value = True
        mock_monitor.interrupted_event.is_set.return_value = False
        mock_monitor.cancelled_event.is_set.return_value = False
        mock_monitor_cls.return_value = mock_monitor

        engine = PresentationEngine(session, settings, prepare_dir=prepare_dir)
        result = await engine._wait_for_start(0, 3)

        assert result is None
        mock_monitor.start.assert_called_once()
        mock_monitor.stop.assert_called_once()

    @patch("neev_voice.audio.keyboard.KeyboardMonitor")
    @patch("rich.console.Console")
    async def test_space_pressed_returns_interrupted(
        self,
        mock_console_cls: MagicMock,
        mock_monitor_cls: MagicMock,
        session: SessionInfo,
        settings: MagicMock,
        prepare_dir: Path,
    ) -> None:
        """SPACE press returns interrupted with correct state_data."""
        mock_monitor = MagicMock()
        mock_monitor.done_event.is_set.return_value = False
        mock_monitor.interrupted_event.is_set.return_value = True
        mock_monitor.cancelled_event.is_set.return_value = False
        mock_monitor_cls.return_value = mock_monitor

        engine = PresentationEngine(session, settings, prepare_dir=prepare_dir)
        result = await engine._wait_for_start(2, 5)

        assert result is not None
        assert result.interrupted is True
        assert result.state_data["current_concept_index"] == 2
        assert result.state_data["total_concepts"] == 5

    @patch("neev_voice.audio.keyboard.KeyboardMonitor")
    @patch("rich.console.Console")
    async def test_esc_pressed_returns_cancelled(
        self,
        mock_console_cls: MagicMock,
        mock_monitor_cls: MagicMock,
        session: SessionInfo,
        settings: MagicMock,
        prepare_dir: Path,
    ) -> None:
        """ESC press returns cancelled."""
        mock_monitor = MagicMock()
        mock_monitor.done_event.is_set.return_value = False
        mock_monitor.interrupted_event.is_set.return_value = False
        mock_monitor.cancelled_event.is_set.return_value = True
        mock_monitor_cls.return_value = mock_monitor

        engine = PresentationEngine(session, settings, prepare_dir=prepare_dir)
        result = await engine._wait_for_start(0, 1)

        assert result is not None
        assert result.cancelled is True

    @patch("neev_voice.audio.keyboard.KeyboardMonitor")
    @patch("rich.console.Console")
    async def test_monitor_stopped_on_exception(
        self,
        mock_console_cls: MagicMock,
        mock_monitor_cls: MagicMock,
        session: SessionInfo,
        settings: MagicMock,
        prepare_dir: Path,
    ) -> None:
        """Monitor is stopped even if an exception occurs."""
        mock_monitor = MagicMock()
        mock_monitor.done_event.is_set.side_effect = RuntimeError("test error")
        mock_monitor_cls.return_value = mock_monitor

        engine = PresentationEngine(session, settings, prepare_dir=prepare_dir)

        with pytest.raises(RuntimeError, match="test error"):
            await engine._wait_for_start(0, 1)

        mock_monitor.stop.assert_called_once()

    @patch("neev_voice.audio.keyboard.KeyboardMonitor")
    @patch("rich.console.Console")
    async def test_console_displays_prompt(
        self,
        mock_console_cls: MagicMock,
        mock_monitor_cls: MagicMock,
        session: SessionInfo,
        settings: MagicMock,
        prepare_dir: Path,
    ) -> None:
        """Console prints the key instructions prompt."""
        mock_console = MagicMock()
        mock_console_cls.return_value = mock_console

        mock_monitor = MagicMock()
        mock_monitor.done_event.is_set.return_value = True
        mock_monitor_cls.return_value = mock_monitor

        engine = PresentationEngine(session, settings, prepare_dir=prepare_dir)
        await engine._wait_for_start(1, 3)

        mock_console.print.assert_called_once()
        call_args = mock_console.print.call_args[0][0]
        assert "2/3" in call_args
        assert "ENTER" in call_args
        assert "SPACE" in call_args
        assert "ESC" in call_args

    @patch("neev_voice.audio.keyboard.KeyboardMonitor")
    @patch("rich.console.Console")
    async def test_uses_presentation_mode(
        self,
        mock_console_cls: MagicMock,
        mock_monitor_cls: MagicMock,
        session: SessionInfo,
        settings: MagicMock,
        prepare_dir: Path,
    ) -> None:
        """KeyboardMonitor is created with PRESENTATION mode."""
        from neev_voice.audio.keyboard import MonitorMode

        mock_monitor = MagicMock()
        mock_monitor.done_event.is_set.return_value = True
        mock_monitor_cls.return_value = mock_monitor

        engine = PresentationEngine(session, settings, prepare_dir=prepare_dir)
        await engine._wait_for_start(0, 1)

        mock_monitor_cls.assert_called_once_with(mode=MonitorMode.PRESENTATION)


class TestPresentationEngineRun:
    """Tests for run() and run_answer() methods."""

    async def test_run_no_concepts(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """run() returns completed when no concepts exist."""
        engine = PresentationEngine(session, settings, prepare_dir=tmp_path)
        result = await engine.run()
        assert result.completed

    @patch.object(PresentationEngine, "_wait_for_start", new_callable=AsyncMock, return_value=None)
    async def test_run_no_tts_provider(
        self,
        mock_wait: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        prepare_dir: Path,
    ) -> None:
        """run() handles missing TTS provider gracefully."""
        engine = PresentationEngine(session, settings, tts_provider=None, prepare_dir=prepare_dir)
        result = await engine.run()
        assert result.completed
        # Gate called once per concept with a transcript (3 concepts)
        assert mock_wait.call_count == 3

    @patch.object(PresentationEngine, "_wait_for_start", new_callable=AsyncMock, return_value=None)
    async def test_run_tts_error(
        self,
        mock_wait: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        prepare_dir: Path,
    ) -> None:
        """run() handles TTS synthesis errors."""
        tts = AsyncMock()
        tts.synthesize = AsyncMock(side_effect=RuntimeError("TTS failed"))

        engine = PresentationEngine(session, settings, tts_provider=tts, prepare_dir=prepare_dir)
        result = await engine.run()
        assert result.completed

    async def test_run_answer_empty(
        self, session: SessionInfo, settings: MagicMock, prepare_dir: Path
    ) -> None:
        """run_answer() returns completed for empty text."""
        engine = PresentationEngine(session, settings, prepare_dir=prepare_dir)
        result = await engine.run_answer("")
        assert result.completed

    @patch.object(PresentationEngine, "_wait_for_start", new_callable=AsyncMock, return_value=None)
    async def test_run_with_start_index(
        self,
        mock_wait: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        prepare_dir: Path,
    ) -> None:
        """run() respects start_index parameter."""
        engine = PresentationEngine(session, settings, tts_provider=None, prepare_dir=prepare_dir)
        result = await engine.run(start_index=2)
        assert result.completed
        # Only last concept, so gate called once
        mock_wait.assert_called_once_with(2, 3)

    @patch.object(PresentationEngine, "_wait_for_start", new_callable=AsyncMock, return_value=None)
    async def test_run_missing_transcript_skipped(
        self,
        mock_wait: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        prepare_dir: Path,
    ) -> None:
        """run() skips concepts with missing transcripts."""
        for f in (prepare_dir / "transcripts").iterdir():
            if f.name.startswith("001_"):
                f.unlink()

        engine = PresentationEngine(session, settings, tts_provider=None, prepare_dir=prepare_dir)
        result = await engine.run()
        assert result.completed
        # Gate called for concepts 0 and 2 (concept 1 has no transcript, skipped before gate)
        assert mock_wait.call_count == 2

    @patch.object(PresentationEngine, "_wait_for_start", new_callable=AsyncMock)
    async def test_run_gate_interrupted_at_first_concept(
        self,
        mock_wait: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        prepare_dir: Path,
    ) -> None:
        """SPACE during wait gate returns interrupted without starting TTS."""
        mock_wait.return_value = PresentationResult(
            interrupted=True,
            state_data={"current_concept_index": 0, "total_concepts": 3},
        )

        engine = PresentationEngine(session, settings, tts_provider=None, prepare_dir=prepare_dir)
        result = await engine.run()

        assert result.interrupted
        assert result.state_data["current_concept_index"] == 0
        mock_wait.assert_called_once_with(0, 3)

    @patch.object(PresentationEngine, "_wait_for_start", new_callable=AsyncMock)
    async def test_run_gate_cancelled(
        self,
        mock_wait: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        prepare_dir: Path,
    ) -> None:
        """ESC during wait gate returns cancelled."""
        mock_wait.return_value = PresentationResult(cancelled=True)

        engine = PresentationEngine(session, settings, tts_provider=None, prepare_dir=prepare_dir)
        result = await engine.run()

        assert result.cancelled

    @patch.object(PresentationEngine, "_wait_for_start", new_callable=AsyncMock)
    async def test_run_gate_interrupted_at_second_concept(
        self,
        mock_wait: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        prepare_dir: Path,
    ) -> None:
        """SPACE during second concept gate returns interrupted with correct index."""
        mock_wait.side_effect = [
            None,  # First concept: ENTER pressed, proceed
            PresentationResult(
                interrupted=True,
                state_data={"current_concept_index": 1, "total_concepts": 3},
            ),
        ]

        engine = PresentationEngine(session, settings, tts_provider=None, prepare_dir=prepare_dir)
        result = await engine.run()

        assert result.interrupted
        assert result.state_data["current_concept_index"] == 1
        assert mock_wait.call_count == 2

    async def test_run_answer_skips_gate(
        self,
        session: SessionInfo,
        settings: MagicMock,
        prepare_dir: Path,
    ) -> None:
        """run_answer() starts TTS immediately without ENTER gate."""
        engine = PresentationEngine(session, settings, tts_provider=None, prepare_dir=prepare_dir)
        result = await engine.run_answer("Some answer text")
        # No TTS provider → no audio → completed=False (not blocked on gate)
        assert not result.completed

    @patch.object(PresentationEngine, "_wait_for_start", new_callable=AsyncMock, return_value=None)
    async def test_on_concept_done_called_for_each_concept(
        self,
        mock_wait: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        prepare_dir: Path,
    ) -> None:
        """on_concept_done callback is called after each concept completes."""
        cb = MagicMock()
        engine = PresentationEngine(
            session, settings, tts_provider=None, prepare_dir=prepare_dir, on_concept_done=cb
        )
        result = await engine.run()

        assert result.completed
        assert cb.call_count == 3
        cb.assert_any_call(0)
        cb.assert_any_call(1)
        cb.assert_any_call(2)

    @patch.object(PresentationEngine, "_wait_for_start", new_callable=AsyncMock)
    async def test_on_concept_done_not_called_on_interrupt(
        self,
        mock_wait: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        prepare_dir: Path,
    ) -> None:
        """on_concept_done is not called when gate returns interrupted."""
        mock_wait.return_value = PresentationResult(
            interrupted=True,
            state_data={"current_concept_index": 0, "total_concepts": 3},
        )
        cb = MagicMock()
        engine = PresentationEngine(
            session, settings, tts_provider=None, prepare_dir=prepare_dir, on_concept_done=cb
        )
        await engine.run()

        cb.assert_not_called()

    @patch.object(PresentationEngine, "_wait_for_start", new_callable=AsyncMock, return_value=None)
    async def test_on_concept_done_none_callback_no_error(
        self,
        mock_wait: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        prepare_dir: Path,
    ) -> None:
        """No error when on_concept_done is None (default)."""
        engine = PresentationEngine(session, settings, tts_provider=None, prepare_dir=prepare_dir)
        result = await engine.run()
        assert result.completed
