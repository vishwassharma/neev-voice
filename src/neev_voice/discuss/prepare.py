"""Prepare engine for the discuss state machine.

Orchestrates the research phase using Claude CLI to analyze input
documents and generate tutorials, explainers, TTS-ready transcripts,
and concept indexes for progressive understanding.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from neev_voice.discuss.session import SessionInfo

if TYPE_CHECKING:
    from neev_voice.config import NeevSettings

logger = structlog.get_logger(__name__)

__all__ = ["ConceptInfo", "PrepareEngine"]


@dataclass
class ConceptInfo:
    """Metadata for a single extracted concept.

    Attributes:
        index: Zero-based concept order in the learning sequence.
        title: Short concept title.
        description: One-line description of the concept.
        source_file: Input document from which this concept was extracted.
        dependencies: Indices of prerequisite concepts.
    """

    index: int
    title: str
    description: str
    source_file: str = ""
    dependencies: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConceptInfo:
        """Deserialize from dictionary.

        Args:
            data: Dictionary with concept fields.

        Returns:
            Reconstructed ConceptInfo.
        """
        return cls(**data)


PREPARE_PLAN_PROMPT = """\
You are analyzing research documents for an interactive learning system.

Given the following document, perform a detailed structural analysis:

1. Identify the key concepts explained in this document
2. Order them from foundational to advanced (progressive learning)
3. For each concept, note its dependencies on other concepts
4. Note which parts explain individual concepts vs. how concepts integrate

Output your analysis as JSON with this structure:
{{
  "concepts": [
    {{
      "index": 0,
      "title": "Concept Name",
      "description": "One-line description",
      "source_file": "{source_file}",
      "dependencies": []
    }}
  ]
}}

Document path: {doc_path}
Document content follows:
---
{content}
---
"""

PREPARE_TUTORIAL_PROMPT = """\
You are generating educational content for an interactive learning system.

Based on the following concept from the research documents, generate:

1. A **tutorial** (detailed explanation with examples, suitable for reading)
2. An **explainer** (brief 2-3 paragraph overview)
3. A **transcript** (TTS-ready text optimized for speech synthesis — use natural
   spoken language, avoid markdown formatting, spell out abbreviations)

Context: This concept is part of a progressive learning sequence.
Previously covered concepts: {previous_concepts}

Concept: {concept_title}
Description: {concept_description}
Source document: {source_file}

Relevant source content:
---
{relevant_content}
---

Output your response with these exact sections:
## Tutorial
[detailed tutorial content]

## Explainer
[brief explainer content]

## Transcript
[TTS-ready spoken text]
"""


class PrepareEngine:
    """Orchestrates the research preparation phase.

    Scans the research_path folder for input documents, uses Claude CLI
    to analyze each file, and generates tutorials, explainers, TTS-ready
    transcripts, and concept indexes for progressive understanding.

    Attributes:
        session: Current discuss session info.
        settings: Application settings.
        prepare_dir: Path to the prepare phase output directory.
    """

    def __init__(
        self,
        session: SessionInfo,
        settings: NeevSettings,
        prepare_dir: Path | None = None,
    ) -> None:
        """Initialize the prepare engine.

        Args:
            session: Current discuss session info.
            settings: Application settings.
            prepare_dir: Override prepare output directory.
        """
        self.session = session
        self.settings = settings
        self.prepare_dir = prepare_dir or Path(settings.discuss_base_dir) / session.name / "prepare"

    async def run(self) -> list[ConceptInfo]:
        """Run the full prepare phase.

        1. Check for existing concepts.json (skip Phase 1 if present)
        2. Scan research_path for documents
        3. Analyze each document for concepts
        4. Generate tutorials, explainers, transcripts (skip existing)
        5. Save concepts.json and output files

        Returns:
            Ordered list of extracted ConceptInfo objects.
        """
        logger.info("prepare_engine_started", session=self.session.name)

        # Setup output directories
        self._ensure_dirs()

        # Check for existing concepts (Phase 1 resume)
        all_concepts = self._load_existing_concepts()
        if all_concepts:
            logger.info("prepare_phase1_complete", existing_concepts=len(all_concepts))
        else:
            # Phase 1: Extract concepts from all documents
            docs = self._find_documents()
            if not docs:
                logger.warning("prepare_engine_no_documents", path=self.session.research_path)
                return []

            logger.info("prepare_engine_found_documents", count=len(docs))

            all_concepts = []
            for doc in docs:
                concepts = await self._extract_concepts(doc)
                # Re-index concepts to be globally sequential
                offset = len(all_concepts)
                for c in concepts:
                    c.index = offset + c.index
                    c.dependencies = [d + offset for d in c.dependencies]
                all_concepts.extend(concepts)

            if not all_concepts:
                logger.warning("prepare_engine_no_concepts_extracted")
                return []

            # Save concept index
            self._save_concepts(all_concepts)

        # Phase 2: Generate content for each concept (skip existing)
        pending = [c for c in all_concepts if not self._concept_content_exists(c.index)]
        if pending:
            logger.info(
                "prepare_phase2_progress",
                total=len(all_concepts),
                pending=len(pending),
                skipped=len(all_concepts) - len(pending),
            )
            for concept in pending:
                previous = [c.title for c in all_concepts[: concept.index]]
                await self._generate_content(concept, previous)
        else:
            logger.info("prepare_phase2_complete", total=len(all_concepts))

        logger.info(
            "prepare_engine_complete",
            total_concepts=len(all_concepts),
        )
        return all_concepts

    def _ensure_dirs(self) -> None:
        """Create output directory structure."""
        for subdir in ("tutorials", "explainers", "transcripts", "indexes"):
            (self.prepare_dir / subdir).mkdir(parents=True, exist_ok=True)

    def _find_documents(self) -> list[Path]:
        """Find all readable documents in the research path.

        Returns:
            Sorted list of document file paths.
        """
        research_path = Path(self.session.research_path)
        if not research_path.exists():
            return []

        extensions = self.settings.resolved_doc_extensions
        return [
            f
            for f in sorted(research_path.rglob("*"))
            if f.is_file() and f.suffix.lower() in extensions
        ]

    def _load_existing_concepts(self) -> list[ConceptInfo] | None:
        """Load previously extracted concepts if they exist.

        Returns:
            List of ConceptInfo if concepts.json exists, None otherwise.
        """
        concepts_file = self.prepare_dir / "concepts.json"
        if not concepts_file.exists():
            return None
        try:
            data = json.loads(concepts_file.read_text(encoding="utf-8"))
            return [ConceptInfo.from_dict(c) for c in data]
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def _concept_content_exists(self, concept_index: int) -> bool:
        """Check if all content artifacts exist for a concept.

        Checks for the presence of tutorial, explainer, and transcript
        files matching the concept index prefix.

        Args:
            concept_index: Zero-based concept index.

        Returns:
            True if all three content files exist.
        """
        prefix = f"{concept_index:03d}_"
        for subdir in ("tutorials", "explainers", "transcripts"):
            dir_path = self.prepare_dir / subdir
            if not dir_path.exists():
                return False
            matches = [f for f in dir_path.iterdir() if f.name.startswith(prefix) and f.is_file()]
            if not matches:
                return False
        return True

    def _save_concepts(self, concepts: list[ConceptInfo]) -> None:
        """Save the concept index to concepts.json.

        Args:
            concepts: Ordered list of concepts to save.
        """
        concepts_file = self.prepare_dir / "concepts.json"
        concepts_file.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps([c.to_dict() for c in concepts], indent=2, default=str)
        concepts_file.write_text(content + "\n", encoding="utf-8")

    async def _extract_concepts(self, doc_path: Path) -> list[ConceptInfo]:
        """Extract concepts from a single document using Claude CLI.

        Args:
            doc_path: Path to the input document.

        Returns:
            List of extracted concepts from this document.
        """
        try:
            content = doc_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("prepare_skip_binary_file", path=str(doc_path))
            return []

        # Truncate very large documents
        max_chars = self.settings.discuss_max_doc_chars
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n[... truncated ...]"

        prompt = PREPARE_PLAN_PROMPT.format(
            doc_path=str(doc_path),
            source_file=doc_path.name,
            content=content,
        )

        response = await self._run_claude(prompt)
        return self._parse_concepts_response(response, doc_path.name)

    def _parse_concepts_response(self, response: str, source_file: str) -> list[ConceptInfo]:
        """Parse Claude's concept extraction response.

        Attempts to extract JSON from the response. Falls back to
        creating a single concept if JSON parsing fails.

        Args:
            response: Raw Claude CLI output.
            source_file: Name of the source document.

        Returns:
            List of parsed ConceptInfo objects.
        """
        # Try to find JSON in the response
        import re

        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            try:
                data = json.loads(json_match.group())
                concepts_data = data.get("concepts", [])
                return [
                    ConceptInfo(
                        index=c.get("index", i),
                        title=c.get("title", f"Concept {i}"),
                        description=c.get("description", ""),
                        source_file=c.get("source_file", source_file),
                        dependencies=c.get("dependencies", []),
                    )
                    for i, c in enumerate(concepts_data)
                ]
            except (json.JSONDecodeError, TypeError):
                pass

        # Fallback: create a single concept for the whole document
        logger.warning("prepare_concepts_parse_fallback", source_file=source_file)
        return [
            ConceptInfo(
                index=0,
                title=source_file,
                description=f"Content from {source_file}",
                source_file=source_file,
            )
        ]

    async def _generate_content(self, concept: ConceptInfo, previous_concepts: list[str]) -> None:
        """Generate tutorial, explainer, and transcript for a concept.

        Args:
            concept: The concept to generate content for.
            previous_concepts: Titles of previously covered concepts.
        """
        # Read relevant source content
        source_path = Path(self.session.research_path) / concept.source_file
        relevant_content = ""
        if source_path.exists():
            try:
                relevant_content = source_path.read_text(encoding="utf-8")
                max_source = self.settings.discuss_max_source_chars
                if len(relevant_content) > max_source:
                    relevant_content = relevant_content[:max_source] + "\n[... truncated ...]"
            except UnicodeDecodeError:
                relevant_content = "[binary file — not readable]"

        prompt = PREPARE_TUTORIAL_PROMPT.format(
            concept_title=concept.title,
            concept_description=concept.description,
            source_file=concept.source_file,
            previous_concepts=", ".join(previous_concepts) if previous_concepts else "none",
            relevant_content=relevant_content,
        )

        response = await self._run_claude(prompt)
        self._save_content(concept, response)

    def _save_content(self, concept: ConceptInfo, response: str) -> None:
        """Parse and save tutorial, explainer, and transcript from response.

        Args:
            concept: The concept this content is for.
            response: Claude CLI response with ## Tutorial, ## Explainer, ## Transcript sections.
        """
        import re

        prefix = f"{concept.index:03d}_{_slugify(concept.title)}"

        sections: dict[str, str] = {}
        current_section: str | None = None
        current_lines: list[str] = []

        for line in response.split("\n"):
            header_match = re.match(r"^##\s+(.+)$", line)
            if header_match:
                if current_section:
                    sections[current_section] = "\n".join(current_lines).strip()
                current_section = header_match.group(1).strip().lower()
                current_lines = []
            else:
                current_lines.append(line)

        if current_section:
            sections[current_section] = "\n".join(current_lines).strip()

        # Save each type
        mapping = {
            "tutorial": "tutorials",
            "explainer": "explainers",
            "transcript": "transcripts",
        }
        for section_name, dir_name in mapping.items():
            content = sections.get(section_name, "")
            if not content:
                content = response  # fallback: save full response
            path = self.prepare_dir / dir_name / f"{prefix}.md"
            path.write_text(content + "\n", encoding="utf-8")

        logger.info(
            "prepare_content_saved",
            concept=concept.title,
            index=concept.index,
        )

    async def _run_claude(self, prompt: str) -> str:
        """Run a prompt through Claude CLI.

        Uses ``claude --dangerously-skip-permissions`` with MCP config
        and the session name for conversation continuity.

        Args:
            prompt: The prompt text to send.

        Returns:
            Claude CLI stdout as a string.
        """
        mcp_config = self.settings.resolved_mcp_config

        env = dict(os.environ)
        api_key = self.settings.resolved_llm_api_key
        if api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        if self.settings.resolved_llm_api_base:
            env["ANTHROPIC_BASE_URL"] = self.settings.resolved_llm_api_base

        cmd = [
            "claude",
            "--dangerously-skip-permissions",
            "--model",
            self.settings.resolved_discuss_model,
            "--output-format",
            "text",
            "--mcp-config",
            mcp_config,
            "-p",
            prompt,
        ]

        logger.debug("prepare_claude_invocation", cmd_length=len(prompt))

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.warning(
                "prepare_claude_error",
                returncode=process.returncode,
                stderr=stderr.decode("utf-8", errors="replace")[:500],
            )

        return stdout.decode("utf-8")


def _slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug.

    Args:
        text: Input text to slugify.

    Returns:
        Lowercase alphanumeric string with hyphens.
    """
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return slug.strip("-")[:50]
