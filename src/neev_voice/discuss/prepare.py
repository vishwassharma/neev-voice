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

Read and analyze ALL the following documents to extract a unified set of concepts:

{doc_list}

Requirements:
1. Read each document listed above using the Read tool
2. Extract ALL key concepts across ALL documents
3. Categorize related concepts together — group by theme/topic
4. Order from foundational to advanced (progressive learning sequence)
5. Identify dependencies between concepts (which must be understood first)
6. Each concept must reference the source document it came from
7. Avoid duplicate or overlapping concepts — merge related ideas

Output your analysis as a single JSON object with this exact structure:
{{
  "concepts": [
    {{
      "index": 0,
      "title": "Concept Name",
      "description": "One-line description of what this concept covers",
      "source_file": "filename.md",
      "dependencies": []
    }}
  ]
}}

Important: Output ONLY the JSON object, no other text.
"""

PREPARE_CONTENT_PROMPT = """\
You are generating educational content for an interactive learning system.

The concept index is at: {concepts_json_path}
Research documents are at: {research_path}

For each concept listed below, generate THREE files in the output directory:

1. {tutorials_dir}/{{prefix}}.md — Detailed tutorial with examples
2. {explainers_dir}/{{prefix}}.md — Brief 2-3 paragraph overview
3. {transcripts_dir}/{{prefix}}.md — TTS-ready spoken text optimized for \
speech synthesis

Where prefix for each concept is shown below.

Concepts to process:
{concepts_list}

Instructions:
- Read the concept index and relevant source documents first
- For each concept, consider previously covered concepts for context
- Write all three files for each concept using the Write tool
- Process concepts using parallel agents where possible for speed
- Each file should contain ONLY the content (no headers or metadata)

Transcript formatting rules (for TTS quality):
- Use natural spoken language, NO markdown formatting
- Spell out abbreviations (API becomes "A P I", HTTP becomes "H T T P")
- Use commas and periods for natural pauses
- Use ellipsis (...) for longer dramatic pauses
- Use exclamation marks for emphasis
- For Hindi-English mixed content: write English words in English script, \
Hindi words in Devanagari script (code-mixing for natural Hinglish output)
- Add conversational fillers where natural ("so", "now", "let's see")
- Use line breaks between paragraphs for paragraph-level pauses
- Keep sentences short and conversational, not academic
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
        2. Extract concepts from ALL documents in a single Claude call
        3. Generate all content in a single Claude call (parallel via agents)

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
            # Phase 1: Extract concepts from ALL documents in one call
            docs = self._find_documents()
            if not docs:
                logger.warning("prepare_engine_no_documents", path=self.session.research_path)
                return []

            logger.info("prepare_engine_found_documents", count=len(docs))

            all_concepts = await self._extract_all_concepts(docs)
            if not all_concepts:
                logger.warning("prepare_engine_no_concepts_extracted")
                return []

            self._save_concepts(all_concepts)

        # Phase 2: Generate content for pending concepts in one call
        pending = [c for c in all_concepts if not self._concept_content_exists(c.index)]
        if pending:
            logger.info(
                "prepare_phase2_progress",
                total=len(all_concepts),
                pending=len(pending),
                skipped=len(all_concepts) - len(pending),
            )
            await self._generate_all_content(pending)
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

    async def _extract_all_concepts(self, docs: list[Path]) -> list[ConceptInfo]:
        """Extract concepts from all documents in a single Claude CLI call.

        Sends all document paths to Claude for unified analysis, producing
        a properly categorized and ordered concept list across all sources.

        Args:
            docs: List of document file paths to analyze.

        Returns:
            Ordered list of extracted ConceptInfo objects.
        """
        doc_list = "\n".join(f"- {doc}" for doc in docs)
        prompt = PREPARE_PLAN_PROMPT.format(doc_list=doc_list)
        response = await self._run_claude(prompt)

        # Parse unified response — source_file comes from Claude's analysis
        return self._parse_concepts_response(response, docs[0].name if docs else "unknown")

    async def _generate_all_content(self, pending: list[ConceptInfo]) -> None:
        """Generate content for all pending concepts in a single Claude CLI call.

        Instructs Claude to read concepts.json and research documents, then
        write tutorial/explainer/transcript files for each pending concept.
        Claude handles parallelism internally via agents.

        Args:
            pending: List of concepts needing content generation.
        """
        concepts_list_parts = []
        for c in pending:
            prefix = f"{c.index:03d}_{_slugify(c.title)}"
            concepts_list_parts.append(
                f'- [{c.index}] "{c.title}" (source: {c.source_file}) → prefix: {prefix}'
            )
        concepts_list = "\n".join(concepts_list_parts)

        prompt = PREPARE_CONTENT_PROMPT.format(
            concepts_json_path=str(self.prepare_dir / "concepts.json"),
            research_path=self.session.research_path,
            tutorials_dir=str(self.prepare_dir / "tutorials"),
            explainers_dir=str(self.prepare_dir / "explainers"),
            transcripts_dir=str(self.prepare_dir / "transcripts"),
            concepts_list=concepts_list,
        )

        await self._run_claude(prompt)

        # Verify files were created
        for c in pending:
            if self._concept_content_exists(c.index):
                logger.info("prepare_content_saved", concept=c.title, index=c.index)
            else:
                logger.warning("prepare_content_missing", concept=c.title, index=c.index)

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
