"""Tests for session portability (export/import)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from neev_voice.discuss.portability import (
    MANIFEST_VERSION,
    _add_directory_to_zip,
    _build_manifest,
    _load_and_remap_session,
    export_session,
    import_session,
)
from neev_voice.discuss.session import SessionInfo, SessionManager
from neev_voice.discuss.state import DiscussState


@pytest.fixture
def tmp_base(tmp_path: Path) -> Path:
    """Temporary base directory for session storage."""
    return tmp_path / "discuss"


@pytest.fixture
def manager(tmp_base: Path) -> SessionManager:
    """SessionManager with temporary base directory."""
    return SessionManager(base_dir=tmp_base)


@pytest.fixture
def session_with_content(manager: SessionManager, tmp_path: Path) -> SessionInfo:
    """Create a session with research docs and prepare artifacts."""
    # Create research docs
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    (research_dir / "doc.md").write_text("# Research Document")
    (research_dir / "notes.txt").write_text("Some notes")

    # Create session
    session = manager.create_session(
        "test-export",
        research_path=str(research_dir),
        source_path=str(tmp_path / "source"),
        output_path=str(manager.session_dir("test-export") / "output"),
    )
    session.state = DiscussState.PRESENTATION
    session.prepare_complete = True
    session.concepts = [{"index": 0, "title": "C1"}]
    manager.save_session(session)

    # Create prepare artifacts
    session_dir = manager.session_dir("test-export")
    prepare_dir = session_dir / "prepare"
    (prepare_dir / "transcripts").mkdir(parents=True)
    (prepare_dir / "concepts.json").write_text(json.dumps([{"index": 0, "title": "C1"}]))
    (prepare_dir / "transcripts" / "000_c1.md").write_text("Transcript for C1")

    # Create output directory
    output_dir = session_dir / "output"
    output_dir.mkdir(parents=True)
    (output_dir / "result.md").write_text("Some output")

    return session


class TestBuildManifest:
    """Tests for _build_manifest."""

    def test_contains_required_fields(self) -> None:
        """Manifest has all required fields."""
        session = SessionInfo(name="test", research_path="/r", source_path="/s", output_path="/o")
        manifest = _build_manifest(session)

        assert manifest["manifest_version"] == MANIFEST_VERSION
        assert manifest["session_name"] == "test"
        assert "exported_at" in manifest
        assert "neev_voice_version" in manifest
        assert manifest["original_paths"]["research_path"] == "/r"
        assert manifest["original_paths"]["source_path"] == "/s"
        assert manifest["original_paths"]["output_path"] == "/o"


class TestAddDirectoryToZip:
    """Tests for _add_directory_to_zip."""

    def test_adds_files_recursively(self, tmp_path: Path) -> None:
        """Adds all files from directory with correct prefix."""
        src_dir = tmp_path / "src"
        (src_dir / "sub").mkdir(parents=True)
        (src_dir / "a.txt").write_text("aaa")
        (src_dir / "sub" / "b.txt").write_text("bbb")

        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            _add_directory_to_zip(zf, src_dir, "prefix")

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert "prefix/a.txt" in names
            assert "prefix/sub/b.txt" in names

    def test_nonexistent_dir_is_noop(self, tmp_path: Path) -> None:
        """Nonexistent directory is silently skipped."""
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            _add_directory_to_zip(zf, tmp_path / "nonexistent", "prefix")

        with zipfile.ZipFile(zip_path, "r") as zf:
            assert zf.namelist() == []


class TestExportSession:
    """Tests for export_session."""

    def test_export_creates_zip(
        self,
        manager: SessionManager,
        session_with_content: SessionInfo,
        tmp_path: Path,
    ) -> None:
        """export_session creates a zip file at the output path."""
        export_dir = tmp_path / "exports"
        zip_path = export_session(manager, "test-export", output_path=export_dir)

        assert zip_path.exists()
        assert zip_path.name == "test-export.zip"
        assert zip_path.parent == export_dir

    def test_export_contains_manifest(
        self,
        manager: SessionManager,
        session_with_content: SessionInfo,
        tmp_path: Path,
    ) -> None:
        """Exported zip contains valid manifest.json."""
        zip_path = export_session(manager, "test-export", output_path=tmp_path / "out")

        with zipfile.ZipFile(zip_path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["session_name"] == "test-export"
            assert manifest["manifest_version"] == MANIFEST_VERSION

    def test_export_contains_session_json(
        self,
        manager: SessionManager,
        session_with_content: SessionInfo,
        tmp_path: Path,
    ) -> None:
        """Exported zip contains session.json."""
        zip_path = export_session(manager, "test-export", output_path=tmp_path / "out")

        with zipfile.ZipFile(zip_path, "r") as zf:
            session_data = json.loads(zf.read("session.json"))
            assert session_data["name"] == "test-export"
            assert session_data["state"] == "presentation"

    def test_export_contains_prepare_artifacts(
        self,
        manager: SessionManager,
        session_with_content: SessionInfo,
        tmp_path: Path,
    ) -> None:
        """Exported zip contains prepare directory."""
        zip_path = export_session(manager, "test-export", output_path=tmp_path / "out")

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert any(n.startswith("prepare/") for n in names)
            assert "prepare/concepts.json" in names
            assert "prepare/transcripts/000_c1.md" in names

    def test_export_contains_research_docs(
        self,
        manager: SessionManager,
        session_with_content: SessionInfo,
        tmp_path: Path,
    ) -> None:
        """Exported zip contains research documents."""
        zip_path = export_session(manager, "test-export", output_path=tmp_path / "out")

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert "research/doc.md" in names
            assert "research/notes.txt" in names

    def test_export_contains_output_artifacts(
        self,
        manager: SessionManager,
        session_with_content: SessionInfo,
        tmp_path: Path,
    ) -> None:
        """Exported zip contains output directory."""
        zip_path = export_session(manager, "test-export", output_path=tmp_path / "out")

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert "output/result.md" in names

    def test_export_nonexistent_session_raises(self, manager: SessionManager) -> None:
        """export_session raises FileNotFoundError for missing session."""
        with pytest.raises(FileNotFoundError, match="ghost"):
            export_session(manager, "ghost")

    def test_export_defaults_to_cwd(
        self,
        manager: SessionManager,
        session_with_content: SessionInfo,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """export_session defaults output_path to cwd."""
        monkeypatch.chdir(tmp_path)
        zip_path = export_session(manager, "test-export")
        assert zip_path.parent == tmp_path


class TestLoadAndRemapSession:
    """Tests for _load_and_remap_session."""

    def test_remaps_paths(self, tmp_path: Path) -> None:
        """Remaps research_path and output_path."""
        session_data = {
            "name": "test",
            "research_path": "/old/research",
            "source_path": "/old/source",
            "output_path": "/old/output",
            "state": "prepare",
        }
        (tmp_path / "session.json").write_text(json.dumps(session_data))

        session = _load_and_remap_session(
            tmp_path,
            research_path="/new/research",
            output_path="/new/output",
        )

        assert session.research_path == "/new/research"
        assert session.output_path == "/new/output"
        assert session.source_path == "/old/source"  # Preserved

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        """Raises ValueError for corrupt session.json."""
        (tmp_path / "session.json").write_text("not json")
        with pytest.raises(ValueError, match=r"Invalid session\.json"):
            _load_and_remap_session(tmp_path, "/r", "/o")


class TestImportSession:
    """Tests for import_session."""

    def test_import_roundtrip(
        self,
        manager: SessionManager,
        session_with_content: SessionInfo,
        tmp_path: Path,
    ) -> None:
        """Export then import recreates the session."""
        # Export
        zip_path = export_session(manager, "test-export", output_path=tmp_path / "exports")

        # Delete original session
        manager.delete_session("test-export")

        # Import
        imported = import_session(manager, zip_path)

        assert imported.name == "test-export"
        assert imported.state == DiscussState.PRESENTATION
        assert imported.prepare_complete is True

        # Verify files exist
        session_dir = manager.session_dir("test-export")
        assert (session_dir / "session.json").exists()
        assert (session_dir / "prepare" / "concepts.json").exists()
        assert (session_dir / "prepare" / "transcripts" / "000_c1.md").exists()

    def test_import_creates_research_dir(
        self,
        manager: SessionManager,
        session_with_content: SessionInfo,
        tmp_path: Path,
    ) -> None:
        """Import creates research directory with documents."""
        zip_path = export_session(manager, "test-export", output_path=tmp_path / "exports")
        manager.delete_session("test-export")

        imported = import_session(manager, zip_path)

        research_dir = Path(imported.research_path)
        assert research_dir.exists()
        assert (research_dir / "doc.md").exists()
        assert (research_dir / "doc.md").read_text() == "# Research Document"

    def test_import_custom_research_dest(
        self,
        manager: SessionManager,
        session_with_content: SessionInfo,
        tmp_path: Path,
    ) -> None:
        """Import places research docs at custom destination."""
        zip_path = export_session(manager, "test-export", output_path=tmp_path / "exports")
        manager.delete_session("test-export")

        custom_research = tmp_path / "my_research"
        imported = import_session(manager, zip_path, research_dest=custom_research)

        assert imported.research_path == str(custom_research)
        assert (custom_research / "doc.md").exists()

    def test_import_nonexistent_zip_raises(self, manager: SessionManager) -> None:
        """import_session raises FileNotFoundError for missing zip."""
        with pytest.raises(FileNotFoundError, match="not found"):
            import_session(manager, Path("/nonexistent.zip"))

    def test_import_invalid_zip_raises(self, manager: SessionManager, tmp_path: Path) -> None:
        """import_session raises ValueError for zip without manifest."""
        zip_path = tmp_path / "bad.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("random.txt", "hello")

        with pytest.raises(ValueError, match="missing manifest"):
            import_session(manager, zip_path)

    def test_import_duplicate_session_raises(
        self,
        manager: SessionManager,
        session_with_content: SessionInfo,
        tmp_path: Path,
    ) -> None:
        """import_session raises FileExistsError for existing session."""
        zip_path = export_session(manager, "test-export", output_path=tmp_path / "exports")

        # Session still exists — import should fail
        with pytest.raises(FileExistsError, match="already exists"):
            import_session(manager, zip_path)

    def test_import_remaps_paths(
        self,
        manager: SessionManager,
        session_with_content: SessionInfo,
        tmp_path: Path,
    ) -> None:
        """Imported session has local paths, not original machine paths."""
        zip_path = export_session(manager, "test-export", output_path=tmp_path / "exports")
        manager.delete_session("test-export")

        imported = import_session(manager, zip_path)

        session_dir = manager.session_dir("test-export")
        assert imported.output_path == str(session_dir / "output")
        assert imported.research_path == str(session_dir / "research")

    def test_import_session_loadable(
        self,
        manager: SessionManager,
        session_with_content: SessionInfo,
        tmp_path: Path,
    ) -> None:
        """Imported session can be loaded by SessionManager."""
        zip_path = export_session(manager, "test-export", output_path=tmp_path / "exports")
        manager.delete_session("test-export")

        import_session(manager, zip_path)

        loaded = manager.load_session("test-export")
        assert loaded is not None
        assert loaded.name == "test-export"
        assert loaded.state == DiscussState.PRESENTATION
