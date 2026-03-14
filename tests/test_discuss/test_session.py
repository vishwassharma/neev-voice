"""Tests for the discuss session manager."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neev_voice.discuss.migration import CURRENT_SCHEMA_VERSION
from neev_voice.discuss.session import SessionInfo, SessionManager
from neev_voice.discuss.state import DiscussState, StateSnapshot, StateStack


@pytest.fixture
def tmp_base(tmp_path: Path) -> Path:
    """Temporary base directory for session storage."""
    return tmp_path / "discuss"


@pytest.fixture
def manager(tmp_base: Path) -> SessionManager:
    """SessionManager with temporary base directory."""
    return SessionManager(base_dir=tmp_base)


class TestSessionInfo:
    """Tests for SessionInfo dataclass."""

    def test_create_defaults(self) -> None:
        """SessionInfo has sensible defaults."""
        info = SessionInfo(
            name="test-session",
            research_path="/tmp/research",
            source_path="/tmp/source",
            output_path="/tmp/output",
        )
        assert info.state == DiscussState.PREPARE
        assert info.state_stack.is_empty
        assert info.prepare_complete is False
        assert info.presentation_index == 0
        assert info.schema_version == CURRENT_SCHEMA_VERSION
        assert info.concepts is None
        assert info.created_at
        assert info.updated_at

    def test_to_dict(self) -> None:
        """SessionInfo serializes all fields."""
        info = SessionInfo(
            name="my-session",
            research_path="/research",
            source_path="/source",
            output_path="/output",
            state=DiscussState.PRESENTATION,
            prepare_complete=True,
            presentation_index=3,
            concepts=[{"title": "concept1", "index": 0}],
        )
        d = info.to_dict()
        assert d["name"] == "my-session"
        assert d["state"] == "presentation"
        assert d["prepare_complete"] is True
        assert d["presentation_index"] == 3
        assert d["concepts"] == [{"title": "concept1", "index": 0}]
        assert isinstance(d["state_stack"], list)

    def test_from_dict(self) -> None:
        """SessionInfo can be deserialized from dict."""
        d = {
            "name": "restored",
            "research_path": "/r",
            "source_path": "/s",
            "output_path": "/o",
            "state": "enquiry",
            "state_stack": [{"state": "presentation", "data": {"idx": 1}, "timestamp": "t1"}],
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-02T00:00:00",
            "prepare_complete": True,
            "presentation_index": 5,
            "concepts": [{"title": "c1"}],
        }
        info = SessionInfo.from_dict(d)
        assert info.name == "restored"
        assert info.state == DiscussState.ENQUIRY
        assert len(info.state_stack) == 1
        assert info.prepare_complete is True
        assert info.presentation_index == 5
        assert info.concepts == [{"title": "c1"}]

    def test_from_dict_missing_presentation_index_defaults_zero(self) -> None:
        """SessionInfo from_dict defaults presentation_index to 0 for old sessions."""
        d = {
            "name": "old-session",
            "research_path": "/r",
            "source_path": "/s",
            "output_path": "/o",
            "state": "prepare",
        }
        info = SessionInfo.from_dict(d)
        assert info.presentation_index == 0

    def test_from_dict_minimal(self) -> None:
        """SessionInfo from_dict with only required fields."""
        d = {
            "name": "minimal",
            "research_path": "/r",
            "source_path": "/s",
            "output_path": "/o",
            "state": "prepare",
        }
        info = SessionInfo.from_dict(d)
        assert info.name == "minimal"
        assert info.state == DiscussState.PREPARE
        assert info.state_stack.is_empty
        assert info.concepts is None

    def test_from_dict_missing_required_raises(self) -> None:
        """SessionInfo from_dict raises KeyError for missing required fields."""
        with pytest.raises(KeyError):
            SessionInfo.from_dict({"name": "incomplete"})

    def test_roundtrip(self) -> None:
        """SessionInfo survives to_dict/from_dict roundtrip."""
        stack = StateStack()
        stack.push(StateSnapshot(state=DiscussState.PRESENTATION, data={"x": 1}))

        original = SessionInfo(
            name="roundtrip",
            research_path="/research",
            source_path="/source",
            output_path="/output",
            state=DiscussState.ENQUIRY,
            state_stack=stack,
            prepare_complete=True,
            presentation_index=7,
            concepts=[{"title": "test"}],
        )
        restored = SessionInfo.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.state == original.state
        assert len(restored.state_stack) == 1
        assert restored.prepare_complete == original.prepare_complete
        assert restored.presentation_index == original.presentation_index
        assert restored.concepts == original.concepts


class TestSessionManager:
    """Tests for SessionManager CRUD operations."""

    def test_create_session(self, manager: SessionManager, tmp_base: Path) -> None:
        """create_session creates directory and session.json."""
        session = manager.create_session(
            "test-session",
            research_path="/tmp/research",
            source_path="/tmp/source",
        )
        assert session.name == "test-session"
        assert session.state == DiscussState.PREPARE
        assert (tmp_base / "test-session" / "session.json").exists()

    def test_create_session_with_output(self, manager: SessionManager) -> None:
        """create_session respects custom output path."""
        session = manager.create_session(
            "custom-output",
            research_path="/r",
            source_path="/s",
            output_path="/custom/output",
        )
        assert session.output_path == "/custom/output"

    def test_create_session_default_output(self, manager: SessionManager, tmp_base: Path) -> None:
        """create_session uses session_dir/output as default output path."""
        session = manager.create_session("default-out", research_path="/r", source_path="/s")
        expected = str(tmp_base / "default-out" / "output")
        assert session.output_path == expected

    def test_create_duplicate_raises(self, manager: SessionManager) -> None:
        """create_session raises FileExistsError for duplicate name."""
        manager.create_session("dupe", research_path="/r", source_path="/s")
        with pytest.raises(FileExistsError, match="dupe"):
            manager.create_session("dupe", research_path="/r", source_path="/s")

    def test_load_session(self, manager: SessionManager) -> None:
        """load_session returns persisted session."""
        manager.create_session("load-test", research_path="/r", source_path="/s")
        loaded = manager.load_session("load-test")
        assert loaded is not None
        assert loaded.name == "load-test"
        assert loaded.state == DiscussState.PREPARE

    def test_load_nonexistent_returns_none(self, manager: SessionManager) -> None:
        """load_session returns None for missing session."""
        assert manager.load_session("nonexistent") is None

    def test_load_corrupted_returns_none(self, manager: SessionManager, tmp_base: Path) -> None:
        """load_session returns None for corrupted session.json."""
        session_dir = tmp_base / "corrupted"
        session_dir.mkdir(parents=True)
        (session_dir / "session.json").write_text("not json", encoding="utf-8")
        assert manager.load_session("corrupted") is None

    def test_save_session_updates_timestamp(self, manager: SessionManager) -> None:
        """save_session updates the updated_at timestamp."""
        session = manager.create_session("save-test", research_path="/r", source_path="/s")
        original_time = session.updated_at

        session.state = DiscussState.PRESENTATION
        manager.save_session(session)

        loaded = manager.load_session("save-test")
        assert loaded is not None
        assert loaded.state == DiscussState.PRESENTATION
        assert loaded.updated_at >= original_time

    def test_save_session_atomic_write(self, manager: SessionManager, tmp_base: Path) -> None:
        """save_session uses atomic write (no partial files left)."""
        session = manager.create_session("atomic-test", research_path="/r", source_path="/s")
        session.state = DiscussState.ENQUIRY
        manager.save_session(session)

        # No temp files should remain
        session_dir = tmp_base / "atomic-test"
        tmp_files = list(session_dir.glob(".session_*.tmp"))
        assert len(tmp_files) == 0

    def test_save_session_preserves_stack(self, manager: SessionManager) -> None:
        """save_session persists state stack correctly."""
        session = manager.create_session("stack-test", research_path="/r", source_path="/s")
        session.state_stack.push(StateSnapshot(state=DiscussState.PRESENTATION, data={"idx": 3}))
        session.state = DiscussState.ENQUIRY
        manager.save_session(session)

        loaded = manager.load_session("stack-test")
        assert loaded is not None
        assert len(loaded.state_stack) == 1
        assert loaded.state_stack.peek().data == {"idx": 3}

    def test_list_sessions_empty(self, manager: SessionManager) -> None:
        """list_sessions returns empty list when no sessions exist."""
        assert manager.list_sessions() == []

    def test_list_sessions(self, manager: SessionManager) -> None:
        """list_sessions returns sorted session names."""
        manager.create_session("bravo", research_path="/r", source_path="/s")
        manager.create_session("alpha", research_path="/r", source_path="/s")
        manager.create_session("charlie", research_path="/r", source_path="/s")
        assert manager.list_sessions() == ["alpha", "bravo", "charlie"]

    def test_list_sessions_ignores_dirs_without_json(
        self, manager: SessionManager, tmp_base: Path
    ) -> None:
        """list_sessions skips directories without session.json."""
        manager.create_session("valid", research_path="/r", source_path="/s")
        (tmp_base / "invalid-dir").mkdir(parents=True)
        assert manager.list_sessions() == ["valid"]

    def test_get_latest_session_empty(self, manager: SessionManager) -> None:
        """get_latest_session returns None when no sessions exist."""
        assert manager.get_latest_session() is None

    def test_get_latest_session(self, manager: SessionManager) -> None:
        """get_latest_session returns session with latest updated_at."""
        manager.create_session("first", research_path="/r", source_path="/s")
        s2 = manager.create_session("second", research_path="/r", source_path="/s")

        # Force second to have later timestamp
        s2.state = DiscussState.PRESENTATION
        manager.save_session(s2)

        latest = manager.get_latest_session()
        assert latest is not None
        assert latest.name == "second"

    def test_delete_session(self, manager: SessionManager, tmp_base: Path) -> None:
        """delete_session removes directory and returns True."""
        manager.create_session("to-delete", research_path="/r", source_path="/s")
        assert (tmp_base / "to-delete").exists()

        result = manager.delete_session("to-delete")
        assert result is True
        assert not (tmp_base / "to-delete").exists()

    def test_delete_nonexistent_returns_false(self, manager: SessionManager) -> None:
        """delete_session returns False for missing session."""
        assert manager.delete_session("ghost") is False

    def test_session_dir(self, manager: SessionManager, tmp_base: Path) -> None:
        """session_dir returns correct path."""
        assert manager.session_dir("test") == tmp_base / "test"

    def test_session_file(self, manager: SessionManager, tmp_base: Path) -> None:
        """session_file returns correct path."""
        assert manager.session_file("test") == tmp_base / "test" / "session.json"

    def test_output_dir(self, manager: SessionManager) -> None:
        """output_dir returns path from session's output_path."""
        session = SessionInfo(
            name="test", research_path="/r", source_path="/s", output_path="/custom/out"
        )
        assert manager.output_dir(session) == Path("/custom/out")

    def test_session_json_is_valid(self, manager: SessionManager, tmp_base: Path) -> None:
        """Saved session.json is valid JSON."""
        manager.create_session("json-test", research_path="/r", source_path="/s")
        content = (tmp_base / "json-test" / "session.json").read_text(encoding="utf-8")
        data = json.loads(content)
        assert data["name"] == "json-test"
        assert data["state"] == "prepare"

    def test_create_session_with_path_objects(self, manager: SessionManager) -> None:
        """create_session accepts Path objects for paths."""
        session = manager.create_session(
            "path-test",
            research_path=Path("/tmp/research"),
            source_path=Path("/tmp/source"),
            output_path=Path("/tmp/output"),
        )
        assert session.research_path == "/tmp/research"
        assert session.source_path == "/tmp/source"
        assert session.output_path == "/tmp/output"
