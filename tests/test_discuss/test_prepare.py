"""Tests for the prepare engine."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from neev_voice.discuss.prepare import ConceptInfo, PrepareEngine, _slugify
from neev_voice.discuss.session import SessionInfo


@pytest.fixture
def session(tmp_path: Path) -> SessionInfo:
    """Create a test session with research path."""
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    return SessionInfo(
        name="test-session",
        research_path=str(research_dir),
        source_path=str(tmp_path / "source"),
        output_path=str(tmp_path / "output"),
    )


@pytest.fixture
def settings() -> MagicMock:
    """Create mock settings."""
    s = MagicMock()
    s.claude_model = "sonnet"
    s.resolved_llm_api_key = "test-key"
    s.resolved_llm_api_base = ""
    s.resolved_mcp_config = "/mock/mcp.json"
    s.resolved_discuss_model = "sonnet"
    s.resolved_doc_extensions = {".md", ".txt", ".rst", ".html", ".pdf", ".json"}
    s.discuss_max_doc_chars = 100_000
    s.discuss_max_source_chars = 50_000
    s.discuss_base_dir = ".scratch/neev/discuss"
    return s


class TestConceptInfo:
    """Tests for ConceptInfo dataclass."""

    def test_create(self) -> None:
        """ConceptInfo stores all fields."""
        c = ConceptInfo(
            index=0,
            title="Dependency Injection",
            description="DI pattern overview",
            source_file="arch.md",
            dependencies=[],
        )
        assert c.index == 0
        assert c.title == "Dependency Injection"

    def test_to_dict(self) -> None:
        """ConceptInfo serializes to dict."""
        c = ConceptInfo(index=1, title="Test", description="Desc", dependencies=[0])
        d = c.to_dict()
        assert d["index"] == 1
        assert d["dependencies"] == [0]

    def test_from_dict(self) -> None:
        """ConceptInfo deserializes from dict."""
        d = {
            "index": 2,
            "title": "Events",
            "description": "Event system",
            "source_file": "events.md",
            "dependencies": [0, 1],
        }
        c = ConceptInfo.from_dict(d)
        assert c.title == "Events"
        assert c.dependencies == [0, 1]

    def test_roundtrip(self) -> None:
        """ConceptInfo survives to_dict/from_dict roundtrip."""
        original = ConceptInfo(index=0, title="Foo", description="Bar", source_file="f.md")
        restored = ConceptInfo.from_dict(original.to_dict())
        assert restored.title == original.title
        assert restored.index == original.index

    def test_default_fields(self) -> None:
        """ConceptInfo has sensible defaults."""
        c = ConceptInfo(index=0, title="T", description="D")
        assert c.source_file == ""
        assert c.dependencies == []


class TestSlugify:
    """Tests for _slugify helper."""

    def test_basic(self) -> None:
        """Slugifies basic text."""
        assert _slugify("Hello World") == "hello-world"

    def test_special_chars(self) -> None:
        """Strips special characters."""
        assert _slugify("foo/bar:baz!") == "foo-bar-baz"

    def test_max_length(self) -> None:
        """Truncates to 50 characters."""
        result = _slugify("a" * 100)
        assert len(result) <= 50

    def test_strips_leading_trailing_hyphens(self) -> None:
        """Strips leading/trailing hyphens."""
        assert _slugify("--hello--") == "hello"

    def test_empty(self) -> None:
        """Empty input returns empty string."""
        assert _slugify("") == ""


class TestPrepareEngine:
    """Tests for PrepareEngine."""

    def test_init(self, session: SessionInfo, settings: MagicMock) -> None:
        """Engine initializes correctly."""
        engine = PrepareEngine(session, settings)
        assert engine.session is session
        assert engine.settings is settings

    def test_init_custom_dir(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """Engine accepts custom prepare directory."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        assert engine.prepare_dir == tmp_path

    def test_ensure_dirs(self, session: SessionInfo, settings: MagicMock, tmp_path: Path) -> None:
        """_ensure_dirs creates all output subdirectories."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        engine._ensure_dirs()
        for subdir in ("tutorials", "explainers", "transcripts", "indexes"):
            assert (tmp_path / subdir).exists()

    def test_find_documents_empty(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """_find_documents returns empty for empty research path."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        assert engine._find_documents() == []

    def test_find_documents(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """_find_documents finds supported file types."""
        research_dir = Path(session.research_path)
        (research_dir / "doc.md").write_text("# Hello")
        (research_dir / "notes.txt").write_text("notes")
        (research_dir / "image.png").write_bytes(b"fake png")

        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        docs = engine._find_documents()
        assert len(docs) == 2
        names = {d.name for d in docs}
        assert "doc.md" in names
        assert "notes.txt" in names
        assert "image.png" not in names

    def test_find_documents_nonexistent_path(self, settings: MagicMock, tmp_path: Path) -> None:
        """_find_documents returns empty for nonexistent path."""
        session = SessionInfo(
            name="t", research_path="/nonexistent", source_path="/s", output_path="/o"
        )
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        assert engine._find_documents() == []

    def test_load_existing_concepts_none(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """_load_existing_concepts returns None when no file exists."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        assert engine._load_existing_concepts() is None

    def test_load_existing_concepts(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """_load_existing_concepts loads from concepts.json."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        tmp_path.mkdir(exist_ok=True)
        concepts_file = tmp_path / "concepts.json"
        concepts_file.write_text(json.dumps([{"index": 0, "title": "C1", "description": "D1"}]))
        result = engine._load_existing_concepts()
        assert result is not None
        assert len(result) == 1
        assert result[0].title == "C1"

    def test_load_existing_concepts_corrupted(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """_load_existing_concepts returns None for corrupted file."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        concepts_file = tmp_path / "concepts.json"
        concepts_file.write_text("not json")
        assert engine._load_existing_concepts() is None

    def test_save_concepts(self, session: SessionInfo, settings: MagicMock, tmp_path: Path) -> None:
        """_save_concepts writes concepts.json."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        concepts = [
            ConceptInfo(index=0, title="A", description="aa"),
            ConceptInfo(index=1, title="B", description="bb"),
        ]
        engine._save_concepts(concepts)

        data = json.loads((tmp_path / "concepts.json").read_text())
        assert len(data) == 2
        assert data[0]["title"] == "A"

    def test_parse_concepts_response_valid_json(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """_parse_concepts_response extracts concepts from JSON."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        response = """Some text before
{
  "concepts": [
    {"index": 0, "title": "Basics", "description": "Basic concepts", "dependencies": []},
    {"index": 1, "title": "Advanced", "description": "Advanced topics", "dependencies": [0]}
  ]
}
Some text after"""
        concepts = engine._parse_concepts_response(response, "test.md")
        assert len(concepts) == 2
        assert concepts[0].title == "Basics"
        assert concepts[1].dependencies == [0]

    def test_parse_concepts_response_no_json_fallback(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """_parse_concepts_response falls back to single concept."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        response = "No JSON here, just text about concepts."
        concepts = engine._parse_concepts_response(response, "doc.md")
        assert len(concepts) == 1
        assert concepts[0].title == "doc.md"

    def test_save_content(self, session: SessionInfo, settings: MagicMock, tmp_path: Path) -> None:
        """_save_content writes tutorial, explainer, and transcript files."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        engine._ensure_dirs()

        concept = ConceptInfo(index=0, title="Test Concept", description="desc")
        response = """## Tutorial
Tutorial content here.

## Explainer
Brief explainer.

## Transcript
This is the spoken version."""
        engine._save_content(concept, response)

        assert (tmp_path / "tutorials" / "000_test-concept.md").exists()
        assert (tmp_path / "explainers" / "000_test-concept.md").exists()
        assert (tmp_path / "transcripts" / "000_test-concept.md").exists()

        tutorial = (tmp_path / "tutorials" / "000_test-concept.md").read_text()
        assert "Tutorial content here." in tutorial

    def test_save_content_fallback(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """_save_content saves full response when sections not found."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        engine._ensure_dirs()

        concept = ConceptInfo(index=0, title="Fallback", description="desc")
        response = "Just plain text without sections."
        engine._save_content(concept, response)

        content = (tmp_path / "tutorials" / "000_fallback.md").read_text()
        assert "Just plain text" in content


class TestConceptContentExists:
    """Tests for _concept_content_exists()."""

    def test_all_files_exist(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """Returns True when tutorial, explainer, and transcript exist."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        engine._ensure_dirs()

        for subdir in ("tutorials", "explainers", "transcripts"):
            (tmp_path / subdir / "000_test-concept.md").write_text("content")

        assert engine._concept_content_exists(0) is True

    def test_missing_transcript(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """Returns False when transcript is missing."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        engine._ensure_dirs()

        (tmp_path / "tutorials" / "000_test.md").write_text("content")
        (tmp_path / "explainers" / "000_test.md").write_text("content")
        # No transcript

        assert engine._concept_content_exists(0) is False

    def test_missing_all(self, session: SessionInfo, settings: MagicMock, tmp_path: Path) -> None:
        """Returns False when no files exist."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        engine._ensure_dirs()
        assert engine._concept_content_exists(0) is False

    def test_no_dirs(self, session: SessionInfo, settings: MagicMock, tmp_path: Path) -> None:
        """Returns False when directories don't exist."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        assert engine._concept_content_exists(0) is False

    def test_different_indices(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """Only matches the correct index prefix."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        engine._ensure_dirs()

        # Create files for index 1, not index 0
        for subdir in ("tutorials", "explainers", "transcripts"):
            (tmp_path / subdir / "001_other.md").write_text("content")

        assert engine._concept_content_exists(0) is False
        assert engine._concept_content_exists(1) is True


class TestExtractAllConcepts:
    """Tests for _extract_all_concepts()."""

    @patch.object(PrepareEngine, "_run_claude")
    async def test_extracts_from_all_docs_in_single_call(
        self,
        mock_claude: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Single Claude call processes all documents."""
        mock_claude.return_value = json.dumps(
            {
                "concepts": [
                    {"index": 0, "title": "A1", "description": "From A", "source_file": "a.md"},
                    {"index": 1, "title": "B1", "description": "From B", "source_file": "b.md"},
                ]
            }
        )

        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        docs = [Path("/docs/a.md"), Path("/docs/b.md")]
        result = await engine._extract_all_concepts(docs)

        assert len(result) == 2
        assert result[0].title == "A1"
        assert result[1].title == "B1"
        # Single call, not one per document
        mock_claude.assert_called_once()
        prompt = mock_claude.call_args[0][0]
        assert "/docs/a.md" in prompt
        assert "/docs/b.md" in prompt


class TestGenerateAllContent:
    """Tests for _generate_all_content()."""

    @patch.object(PrepareEngine, "_run_claude")
    async def test_single_call_for_all_pending(
        self,
        mock_claude: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Single Claude call for all pending concepts."""
        mock_claude.return_value = ""

        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        engine._ensure_dirs()
        pending = [
            ConceptInfo(index=0, title="Concept A", description="Desc A"),
            ConceptInfo(index=1, title="Concept B", description="Desc B"),
        ]
        await engine._generate_all_content(pending)

        mock_claude.assert_called_once()
        prompt = mock_claude.call_args[0][0]
        assert "Concept A" in prompt
        assert "Concept B" in prompt
        assert "000_concept-a" in prompt
        assert "001_concept-b" in prompt

    @patch.object(PrepareEngine, "_run_claude")
    async def test_prompt_includes_output_paths(
        self,
        mock_claude: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Prompt includes correct output directory paths."""
        mock_claude.return_value = ""

        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        engine._ensure_dirs()
        pending = [ConceptInfo(index=0, title="Test", description="D")]
        await engine._generate_all_content(pending)

        prompt = mock_claude.call_args[0][0]
        assert str(tmp_path / "tutorials") in prompt
        assert str(tmp_path / "explainers") in prompt
        assert str(tmp_path / "transcripts") in prompt


class TestPrepareEngineRun:
    """Tests for PrepareEngine.run() method."""

    @patch.object(PrepareEngine, "_generate_all_content")
    async def test_run_resumes_existing_skips_completed(
        self,
        mock_generate: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """run() skips content generation for concepts with existing artifacts."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        engine._ensure_dirs()

        concepts_data = [
            {"index": 0, "title": "Done", "description": "D"},
            {"index": 1, "title": "Pending", "description": "P"},
        ]
        (tmp_path / "concepts.json").write_text(json.dumps(concepts_data))

        # Create all content for concept 0 (already done)
        for subdir in ("tutorials", "explainers", "transcripts"):
            (tmp_path / subdir / "000_done.md").write_text("existing content")

        result = await engine.run()
        assert len(result) == 2

        # Only pending concepts passed to batch generation
        mock_generate.assert_called_once()
        pending_arg = mock_generate.call_args[0][0]
        assert len(pending_arg) == 1
        assert pending_arg[0].index == 1

    @patch.object(PrepareEngine, "_generate_all_content")
    async def test_run_resumes_all_complete(
        self,
        mock_generate: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """run() skips all content generation when everything exists."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        engine._ensure_dirs()

        concepts_data = [{"index": 0, "title": "Done", "description": "D"}]
        (tmp_path / "concepts.json").write_text(json.dumps(concepts_data))

        for subdir in ("tutorials", "explainers", "transcripts"):
            (tmp_path / subdir / "000_done.md").write_text("existing")

        result = await engine.run()
        assert len(result) == 1
        mock_generate.assert_not_called()

    async def test_run_no_documents(
        self, session: SessionInfo, settings: MagicMock, tmp_path: Path
    ) -> None:
        """run() returns empty list when no documents found."""
        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        result = await engine.run()
        assert result == []

    @patch.object(PrepareEngine, "_run_claude")
    async def test_run_single_extraction_call(
        self,
        mock_claude: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """run() uses single Claude call for concept extraction."""
        research_dir = Path(session.research_path)
        (research_dir / "guide.md").write_text("# Guide\nContent here.")

        # Phase 1: extraction (single call)
        # Phase 2: content generation (single call, writes files via Claude)
        extract_response = json.dumps(
            {
                "concepts": [
                    {
                        "index": 0,
                        "title": "Intro",
                        "description": "Introduction",
                        "source_file": "guide.md",
                        "dependencies": [],
                    }
                ]
            }
        )
        # Phase 2 response — Claude writes files directly, response is empty
        mock_claude.side_effect = [extract_response, ""]

        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        result = await engine.run()

        assert len(result) == 1
        assert result[0].title == "Intro"
        assert (tmp_path / "concepts.json").exists()
        # Two Claude calls total: 1 extraction + 1 content generation
        assert mock_claude.call_count == 2

    @patch.object(PrepareEngine, "_run_claude")
    async def test_run_multiple_documents_single_extraction(
        self,
        mock_claude: AsyncMock,
        session: SessionInfo,
        settings: MagicMock,
        tmp_path: Path,
    ) -> None:
        """run() extracts concepts from multiple docs in one call."""
        research_dir = Path(session.research_path)
        (research_dir / "a.md").write_text("Doc A")
        (research_dir / "b.md").write_text("Doc B")

        # Single extraction call returns concepts from both docs
        extract_response = json.dumps(
            {
                "concepts": [
                    {"index": 0, "title": "A1", "description": "From A", "source_file": "a.md"},
                    {"index": 1, "title": "B1", "description": "From B", "source_file": "b.md"},
                ]
            }
        )
        mock_claude.side_effect = [extract_response, ""]

        engine = PrepareEngine(session, settings, prepare_dir=tmp_path)
        result = await engine.run()

        assert len(result) == 2
        assert result[0].index == 0
        assert result[1].index == 1
        # Only 2 calls: 1 extraction + 1 content generation (not 4 as before)
        assert mock_claude.call_count == 2
