"""Tests for the scratch pad module.

Tests ScratchPad initialization, file path properties, save methods,
and class methods for listing and retrieving flow directories.
"""

import json
import re
from pathlib import Path

from neev_voice.scratch import ScratchPad


class TestScratchPadInit:
    """Tests for ScratchPad initialization."""

    def test_creates_flow_directory(self, tmp_path):
        """Test flow directory is created on init."""
        pad = ScratchPad("listen", base_dir=tmp_path)
        assert pad.flow_dir.exists()
        assert pad.flow_dir.is_dir()

    def test_flow_dir_under_base_and_type(self, tmp_path):
        """Test flow directory is under base_dir/flow_type/."""
        pad = ScratchPad("listen", base_dir=tmp_path)
        assert pad.flow_dir.parent == tmp_path / "listen"

    def test_flow_dir_has_timestamp_uuid_name(self, tmp_path):
        """Test flow directory name matches YYYYMMDD-HHMMSS_hexid pattern."""
        pad = ScratchPad("listen", base_dir=tmp_path)
        pattern = r"^\d{8}-\d{6}_[a-f0-9]{8}$"
        assert re.match(pattern, pad.flow_dir.name)

    def test_flow_type_stored(self, tmp_path):
        """Test flow_type attribute is stored correctly."""
        pad = ScratchPad("discussion", base_dir=tmp_path)
        assert pad.flow_type == "discussion"

    def test_base_dir_stored(self, tmp_path):
        """Test base_dir attribute is stored correctly."""
        pad = ScratchPad("listen", base_dir=tmp_path)
        assert pad.base_dir == tmp_path

    def test_default_base_dir_used_when_none(self):
        """Test DEFAULT_BASE_DIR is used when base_dir is None."""
        # Don't actually create directories by checking the attribute
        assert Path(".scratch/neev") == ScratchPad.DEFAULT_BASE_DIR

    def test_unique_dirs_per_instance(self, tmp_path):
        """Test each ScratchPad instance creates a unique directory."""
        pad1 = ScratchPad("listen", base_dir=tmp_path)
        pad2 = ScratchPad("listen", base_dir=tmp_path)
        assert pad1.flow_dir != pad2.flow_dir


class TestScratchPadPaths:
    """Tests for ScratchPad file path properties."""

    def test_audio_path(self, tmp_path):
        """Test audio_path returns audio.wav in flow_dir."""
        pad = ScratchPad("listen", base_dir=tmp_path)
        assert pad.audio_path == pad.flow_dir / "audio.wav"

    def test_transcription_path(self, tmp_path):
        """Test transcription_path returns transcription.txt in flow_dir."""
        pad = ScratchPad("listen", base_dir=tmp_path)
        assert pad.transcription_path == pad.flow_dir / "transcription.txt"

    def test_enriched_path(self, tmp_path):
        """Test enriched_path returns enriched.md in flow_dir."""
        pad = ScratchPad("listen", base_dir=tmp_path)
        assert pad.enriched_path == pad.flow_dir / "enriched.md"

    def test_metadata_path(self, tmp_path):
        """Test metadata_path returns metadata.json in flow_dir."""
        pad = ScratchPad("listen", base_dir=tmp_path)
        assert pad.metadata_path == pad.flow_dir / "metadata.json"


class TestScratchPadDiscussionResultPath:
    """Tests for discussion_result_path property."""

    def test_discussion_result_path(self, tmp_path):
        """Test discussion_result_path returns discussion-result.md in flow_dir."""
        pad = ScratchPad("discussion", base_dir=tmp_path)
        assert pad.discussion_result_path == pad.flow_dir / "discussion-result.md"


class TestScratchPadSaveMethods:
    """Tests for ScratchPad save methods."""

    def test_save_transcription(self, tmp_path):
        """Test save_transcription writes text and returns path."""
        pad = ScratchPad("listen", base_dir=tmp_path)
        path = pad.save_transcription("hello world")
        assert path == pad.transcription_path
        assert path.read_text() == "hello world"

    def test_save_enriched(self, tmp_path):
        """Test save_enriched writes markdown content and returns path."""
        pad = ScratchPad("listen", base_dir=tmp_path)
        content = "# Enrichment Report\n\n## Context\nSome context"
        path = pad.save_enriched(content)
        assert path == pad.enriched_path
        assert path.read_text() == content

    def test_save_metadata(self, tmp_path):
        """Test save_metadata writes JSON with flow info and custom data."""
        pad = ScratchPad("listen", base_dir=tmp_path)
        path = pad.save_metadata(transcription="hello", duration=1.5)
        assert path == pad.metadata_path
        data = json.loads(path.read_text())
        assert data["flow_type"] == "listen"
        assert "timestamp" in data
        assert data["transcription"] == "hello"
        assert data["duration"] == 1.5

    def test_save_metadata_includes_flow_dir(self, tmp_path):
        """Test save_metadata includes the flow_dir in output."""
        pad = ScratchPad("listen", base_dir=tmp_path)
        pad.save_metadata()
        data = json.loads(pad.metadata_path.read_text())
        assert data["flow_dir"] == str(pad.flow_dir)

    def test_save_section(self, tmp_path):
        """Test save_section writes numbered section JSON."""
        pad = ScratchPad("discussion", base_dir=tmp_path)
        data = {"section": "## Intro", "intent": "agreement"}
        path = pad.save_section(1, data)
        assert path == pad.flow_dir / "section_001.json"
        loaded = json.loads(path.read_text())
        assert loaded["section"] == "## Intro"

    def test_save_section_numbering(self, tmp_path):
        """Test save_section uses zero-padded 3-digit numbering."""
        pad = ScratchPad("discussion", base_dir=tmp_path)
        path = pad.save_section(42, {"data": "test"})
        assert path.name == "section_042.json"

    def test_save_summary(self, tmp_path):
        """Test save_summary writes summary.json."""
        pad = ScratchPad("discussion", base_dir=tmp_path)
        data = {"total": 5, "agreements": 3, "disagreements": 2}
        path = pad.save_summary(data)
        assert path == pad.flow_dir / "summary.json"
        loaded = json.loads(path.read_text())
        assert loaded["total"] == 5

    def test_save_discussion_result(self, tmp_path):
        """Test save_discussion_result writes discussion-result.md."""
        pad = ScratchPad("discussion", base_dir=tmp_path)
        content = "# Discussion Result\n\n## Agreements\n- Item 1"
        path = pad.save_discussion_result(content)
        assert path == pad.discussion_result_path
        assert path.read_text() == content

    def test_save_discussion_result_returns_path(self, tmp_path):
        """Test save_discussion_result returns the file path."""
        pad = ScratchPad("discussion", base_dir=tmp_path)
        path = pad.save_discussion_result("test content")
        assert path == pad.flow_dir / "discussion-result.md"

    def test_save_discussion_result_overwrites(self, tmp_path):
        """Test save_discussion_result overwrites existing file."""
        pad = ScratchPad("discussion", base_dir=tmp_path)
        pad.save_discussion_result("old content")
        pad.save_discussion_result("new content")
        assert pad.discussion_result_path.read_text() == "new content"


class TestScratchPadClassMethods:
    """Tests for ScratchPad class methods."""

    def test_get_latest_folder_returns_most_recent(self, tmp_path):
        """Test get_latest_folder returns the newest flow directory."""
        pad1 = ScratchPad("listen", base_dir=tmp_path)
        pad2 = ScratchPad("listen", base_dir=tmp_path)
        latest = ScratchPad.get_latest_folder("listen", base_dir=tmp_path)
        # pad2 was created after pad1, so should be latest
        # (both may have the same second timestamp, so just verify it's one of them)
        assert latest in (pad1.flow_dir, pad2.flow_dir)

    def test_get_latest_folder_returns_none_when_empty(self, tmp_path):
        """Test get_latest_folder returns None when no flows exist."""
        result = ScratchPad.get_latest_folder("listen", base_dir=tmp_path)
        assert result is None

    def test_get_latest_folder_returns_none_for_missing_dir(self, tmp_path):
        """Test get_latest_folder returns None when parent dir doesn't exist."""
        result = ScratchPad.get_latest_folder("listen", base_dir=tmp_path / "nonexistent")
        assert result is None

    def test_list_flows_returns_sorted(self, tmp_path):
        """Test list_flows returns all flow dirs sorted by name."""
        ScratchPad("listen", base_dir=tmp_path)  # creates flow dir 1
        ScratchPad("listen", base_dir=tmp_path)  # creates flow dir 2
        flows = ScratchPad.list_flows("listen", base_dir=tmp_path)
        assert len(flows) == 2
        assert flows == sorted(flows, key=lambda p: p.name)

    def test_list_flows_returns_empty_when_no_flows(self, tmp_path):
        """Test list_flows returns empty list when no flows exist."""
        result = ScratchPad.list_flows("listen", base_dir=tmp_path)
        assert result == []

    def test_list_flows_returns_empty_for_missing_dir(self, tmp_path):
        """Test list_flows returns empty list when parent dir doesn't exist."""
        result = ScratchPad.list_flows("listen", base_dir=tmp_path / "nonexistent")
        assert result == []

    def test_list_flows_ignores_files(self, tmp_path):
        """Test list_flows only returns directories, not files."""
        listen_dir = tmp_path / "listen"
        listen_dir.mkdir(parents=True)
        (listen_dir / "stray_file.txt").write_text("not a dir")
        pad = ScratchPad("listen", base_dir=tmp_path)
        flows = ScratchPad.list_flows("listen", base_dir=tmp_path)
        assert len(flows) == 1
        assert flows[0] == pad.flow_dir
