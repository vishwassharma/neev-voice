"""Tests for session schema migration."""

from __future__ import annotations

import json
from pathlib import Path

from neev_voice.discuss.migration import (
    CURRENT_SCHEMA_VERSION,
    _migrate_v1_to_v2,
    migrate_session_data,
)
from neev_voice.discuss.session import SessionInfo, SessionManager


class TestMigrateV1ToV2:
    """Tests for _migrate_v1_to_v2."""

    def test_adds_presentation_index(self) -> None:
        """Migration adds presentation_index defaulting to 0."""
        data = {"name": "old", "state": "prepare"}
        result = _migrate_v1_to_v2(data)
        assert result["presentation_index"] == 0

    def test_preserves_existing_presentation_index(self) -> None:
        """Migration does not overwrite existing presentation_index."""
        data = {"name": "old", "presentation_index": 5}
        result = _migrate_v1_to_v2(data)
        assert result["presentation_index"] == 5

    def test_sets_schema_version_2(self) -> None:
        """Migration sets schema_version to 2."""
        data = {"name": "old"}
        result = _migrate_v1_to_v2(data)
        assert result["schema_version"] == 2

    def test_preserves_other_fields(self) -> None:
        """Migration preserves all existing fields."""
        data = {
            "name": "test",
            "research_path": "/r",
            "source_path": "/s",
            "output_path": "/o",
            "state": "presentation",
            "prepare_complete": True,
            "concepts": [{"title": "C1"}],
        }
        result = _migrate_v1_to_v2(data)
        assert result["name"] == "test"
        assert result["state"] == "presentation"
        assert result["prepare_complete"] is True
        assert result["concepts"] == [{"title": "C1"}]


class TestMigrateSessionData:
    """Tests for migrate_session_data."""

    def test_v1_to_current(self) -> None:
        """Migrates v1 (no schema_version) to current version."""
        data = {
            "name": "old-session",
            "research_path": "/r",
            "source_path": "/s",
            "output_path": "/o",
            "state": "prepare",
        }
        result, migrated = migrate_session_data(data)
        assert migrated is True
        assert result["schema_version"] == CURRENT_SCHEMA_VERSION
        assert result["presentation_index"] == 0

    def test_already_current_is_noop(self) -> None:
        """Current version returns unchanged with False."""
        data = {
            "name": "current",
            "schema_version": CURRENT_SCHEMA_VERSION,
            "presentation_index": 3,
        }
        result, migrated = migrate_session_data(data)
        assert migrated is False
        assert result is data  # Same object, not copied

    def test_future_version_is_noop(self) -> None:
        """Future schema version returns unchanged with False."""
        data = {"name": "future", "schema_version": 999}
        _result, migrated = migrate_session_data(data)
        assert migrated is False

    def test_missing_schema_version_treated_as_v1(self) -> None:
        """No schema_version key is treated as version 1."""
        data = {"name": "legacy"}
        result, migrated = migrate_session_data(data)
        assert migrated is True
        assert result["schema_version"] == CURRENT_SCHEMA_VERSION


class TestMigrationIntegration:
    """Integration tests for migration via SessionManager."""

    def test_load_session_auto_migrates_v1(self, tmp_path: Path) -> None:
        """Loading a v1 session auto-migrates and persists."""
        base = tmp_path / "discuss"
        mgr = SessionManager(base_dir=base)

        # Write a v1-format session.json (no schema_version, no presentation_index)
        session_dir = base / "legacy-session"
        session_dir.mkdir(parents=True)
        v1_data = {
            "name": "legacy-session",
            "research_path": "/r",
            "source_path": "/s",
            "output_path": "/o",
            "state": "prepare",
            "state_stack": [],
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "prepare_complete": False,
            "concepts": None,
        }
        (session_dir / "session.json").write_text(json.dumps(v1_data))

        # Load should auto-migrate
        session = mgr.load_session("legacy-session")
        assert session is not None
        assert session.schema_version == CURRENT_SCHEMA_VERSION
        assert session.presentation_index == 0

        # Verify persisted with new schema
        raw = json.loads((session_dir / "session.json").read_text())
        assert raw["schema_version"] == CURRENT_SCHEMA_VERSION
        assert raw["presentation_index"] == 0

    def test_load_session_current_no_save(self, tmp_path: Path) -> None:
        """Loading a current-version session does not re-save."""
        base = tmp_path / "discuss"
        mgr = SessionManager(base_dir=base)
        session = mgr.create_session("current", research_path="/r", source_path="/s")

        # Record updated_at
        original_updated = session.updated_at

        # Re-load — should not trigger migration or save
        loaded = mgr.load_session("current")
        assert loaded is not None
        assert loaded.schema_version == CURRENT_SCHEMA_VERSION
        # updated_at unchanged (no save happened)
        assert loaded.updated_at == original_updated

    def test_from_dict_with_schema_version(self) -> None:
        """SessionInfo.from_dict reads schema_version."""
        data = {
            "name": "test",
            "research_path": "/r",
            "source_path": "/s",
            "output_path": "/o",
            "state": "prepare",
            "schema_version": 2,
        }
        info = SessionInfo.from_dict(data)
        assert info.schema_version == 2

    def test_to_dict_includes_schema_version(self) -> None:
        """SessionInfo.to_dict includes schema_version."""
        info = SessionInfo(
            name="test",
            research_path="/r",
            source_path="/s",
            output_path="/o",
        )
        d = info.to_dict()
        assert d["schema_version"] == CURRENT_SCHEMA_VERSION

    def test_roundtrip_preserves_schema_version(self) -> None:
        """Schema version survives to_dict/from_dict roundtrip."""
        info = SessionInfo(
            name="rt",
            research_path="/r",
            source_path="/s",
            output_path="/o",
            schema_version=CURRENT_SCHEMA_VERSION,
        )
        restored = SessionInfo.from_dict(info.to_dict())
        assert restored.schema_version == info.schema_version
