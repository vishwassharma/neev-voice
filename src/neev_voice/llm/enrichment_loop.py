"""Enrichment loop module implementing the Ralph Loop pattern.

Provides the LoopState, IterationResult, and SelfAssessment dataclasses
for managing iterative enrichment agent state, along with functions for
reading/writing state files and parsing structured agent responses.

The Ralph Loop pattern: run an agent iteratively, accumulating state
in external files, with fresh context each iteration. The agent
self-assesses quality and decides when enrichment is complete.

Also provides EnrichmentLoopAgent, the v2 enrichment agent that uses
iterative refinement to produce high-quality enrichment output.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neev_voice.config import NeevSettings

import structlog

logger = structlog.get_logger(__name__)

__all__ = [
    "LOOP_RESPONSE_FORMAT",
    "LOOP_SYSTEM_PROMPT",
    "EnrichmentLoopAgent",
    "IterationResult",
    "LoopState",
    "SelfAssessment",
    "build_iteration_prompt",
    "parse_structured_response",
    "read_state_files",
    "write_state_files",
]

LOOP_SYSTEM_PROMPT = (
    "You are a code-aware analyst performing iterative enrichment. "
    "Given user speech (possibly Hindi-English), explore the codebase to "
    "understand context, then produce a formal, specific enrichment of the "
    "user's problem statement. Reference actual files and functions where "
    "relevant. Do NOT suggest or make code changes.\n\n"
    "You work iteratively: each iteration you explore more, refine your "
    "analysis, and improve the enrichment. You have access to accumulated "
    "state from previous iterations."
)

LOOP_RESPONSE_FORMAT = """\
You MUST structure your response using these exact headings:

## Plan
[Analysis plan — first iteration only, describe what you will explore]

## Thinking
[Current reasoning, what was explored, what was found]

## Memory
[Key discoveries to remember for the next iteration]

## Enrichment

Output your enrichment as structured markdown text (NOT JSON). Use these subsections:

### Summary
[1-2 sentence formal summary of the user's problem statement]

### Key Points
- [Bullet point 1]
- [Bullet point 2]

### Context Analysis
[Describe relevant codebase context discovered via Read/Glob/Grep]

### Relevant Code References
| File | Symbol | Relevance |
|------|--------|-----------|

### Suggested Investigation Areas
- [Area 1]
- [Area 2]

### Atomic Task Breakdown
1. [Task 1]
2. [Task 2]

## Self-Assessment
Quality: N/10
Gaps: [identified gaps or "none"]
Complete: yes/no
"""


@dataclass
class SelfAssessment:
    """Agent's self-assessment of enrichment quality.

    Attributes:
        quality: Quality score from 0-10.
        gaps: Description of identified gaps, or empty string.
        is_complete: Whether the agent considers enrichment complete.
    """

    quality: int = 0
    gaps: str = ""
    is_complete: bool = False


@dataclass
class IterationResult:
    """Parsed result from a single enrichment iteration.

    Each iteration produces structured sections that are parsed
    from the agent's response.

    Attributes:
        plan: Analysis plan (typically set in iteration 1).
        thinking: Current reasoning and exploration notes.
        memory: Key discoveries to remember across iterations.
        enrichment: The enriched output draft.
        self_assessment: Agent's quality self-assessment.
    """

    plan: str = ""
    thinking: str = ""
    memory: str = ""
    enrichment: str = ""
    self_assessment: SelfAssessment = field(default_factory=SelfAssessment)


@dataclass
class LoopState:
    """Accumulated state across enrichment loop iterations.

    Tracks the current iteration number, accumulated plan, thinking,
    draft, and memory content, and whether the loop is complete.

    Attributes:
        iteration: Current iteration number (0 = not started).
        plan: Analysis plan from iteration 1.
        thinking: Current iteration's reasoning (overwritten each time).
        draft: Current best enrichment draft (overwritten each time).
        memory: Accumulated discoveries across all iterations (appended).
        is_complete: Whether the agent has deemed enrichment complete.
    """

    iteration: int = 0
    plan: str = ""
    thinking: str = ""
    draft: str = ""
    memory: str = ""
    is_complete: bool = False

    def update_from_result(self, result: IterationResult, iteration: int) -> LoopState:
        """Create a new LoopState updated with an iteration's result.

        The plan is only updated on the first iteration (or when non-empty
        and no existing plan). Memory is appended across iterations.
        Thinking and draft are overwritten each iteration.

        Args:
            result: Parsed result from the current iteration.
            iteration: The iteration number.

        Returns:
            New LoopState with updated fields.
        """
        # Plan: keep existing unless this is the first or existing is empty
        plan = result.plan if (result.plan and not self.plan) else self.plan

        # Memory: append new discoveries
        if result.memory:
            memory = f"{self.memory}\n\n---\n\n{result.memory}" if self.memory else result.memory
        else:
            memory = self.memory

        return LoopState(
            iteration=iteration,
            plan=plan,
            thinking=result.thinking,
            draft=result.enrichment,
            memory=memory,
            is_complete=result.self_assessment.is_complete,
        )


def write_state_files(state: LoopState, flow_dir: Path) -> None:
    """Write loop state to files in the flow directory.

    Creates the directory if it doesn't exist. Overwrites plan.md,
    thinking.md, enriched_draft.md, and memory.md. Appends to
    loop_log.jsonl.

    Args:
        state: Current loop state to persist.
        flow_dir: Directory to write state files into.
    """
    flow_dir.mkdir(parents=True, exist_ok=True)

    (flow_dir / "plan.md").write_text(state.plan, encoding="utf-8")
    (flow_dir / "thinking.md").write_text(state.thinking, encoding="utf-8")
    (flow_dir / "enriched_draft.md").write_text(state.draft, encoding="utf-8")
    (flow_dir / "memory.md").write_text(state.memory, encoding="utf-8")

    log_entry = {
        "iteration": state.iteration,
        "is_complete": state.is_complete,
        "draft_length": len(state.draft),
        "memory_length": len(state.memory),
    }
    with (flow_dir / "loop_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")

    logger.info(
        "state_files_written",
        flow_dir=str(flow_dir),
        iteration=state.iteration,
    )


def read_state_files(flow_dir: Path) -> LoopState:
    """Read loop state from files in the flow directory.

    Returns a LoopState with whatever files exist. Missing files
    result in empty string fields. Iteration count and is_complete
    are read from the last entry in loop_log.jsonl.

    Args:
        flow_dir: Directory to read state files from.

    Returns:
        LoopState reconstructed from files.
    """
    plan = _read_file(flow_dir / "plan.md")
    thinking = _read_file(flow_dir / "thinking.md")
    draft = _read_file(flow_dir / "enriched_draft.md")
    memory = _read_file(flow_dir / "memory.md")

    iteration = 0
    is_complete = False
    log_path = flow_dir / "loop_log.jsonl"
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        if lines and lines[-1]:
            last_entry = json.loads(lines[-1])
            iteration = last_entry.get("iteration", 0)
            is_complete = last_entry.get("is_complete", False)

    return LoopState(
        iteration=iteration,
        plan=plan,
        thinking=thinking,
        draft=draft,
        memory=memory,
        is_complete=is_complete,
    )


def build_iteration_prompt(
    transcript: str,
    state: LoopState,
    iteration: int,
    max_iterations: int,
) -> str:
    """Build the user prompt for a single enrichment iteration.

    Constructs an iteration-aware prompt that includes the transcript,
    accumulated state from previous iterations, and instructions
    appropriate to the current iteration number.

    Args:
        transcript: The user's transcribed speech.
        state: Current accumulated loop state.
        iteration: The current iteration number (1-based).
        max_iterations: Maximum allowed iterations.

    Returns:
        The assembled prompt string for this iteration.
    """
    parts: list[str] = []

    parts.append(f"**Iteration {iteration} of {max_iterations}**")
    parts.append(f"## Transcript\n{transcript}")

    if iteration == 1:
        parts.append(
            "This is the first iteration. Create an analysis plan, explore "
            "the codebase broadly, and draft an initial enrichment."
        )
    else:
        if state.plan:
            parts.append(f"## Previous Plan\n{state.plan}")
        if state.draft:
            parts.append(f"## Previous Draft\n{state.draft}")
        if state.memory:
            parts.append(f"## Accumulated Memory\n{state.memory}")

        parts.append(
            "Review the previous draft and accumulated memory. Explore "
            "specific files to fill gaps, verify accuracy, and refine "
            "the enrichment."
        )

    parts.append(LOOP_RESPONSE_FORMAT)

    return "\n\n".join(parts)


def _read_file(path: Path) -> str:
    """Read a file's contents, returning empty string if missing.

    Args:
        path: Path to the file.

    Returns:
        File contents or empty string.
    """
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def parse_structured_response(response: str) -> IterationResult:
    """Parse a structured agent response into an IterationResult.

    Expects sections delimited by ``## Plan``, ``## Thinking``,
    ``## Memory``, ``## Enrichment``, and ``## Self-Assessment``.
    Missing sections result in empty strings.

    Args:
        response: Raw agent response text.

    Returns:
        Parsed IterationResult with extracted sections.
    """
    sections = _extract_sections(response)

    plan = sections.get("plan", "").strip()
    thinking = sections.get("thinking", "").strip()
    memory = sections.get("memory", "").strip()
    enrichment = sections.get("enrichment", "").strip()
    self_assessment_text = sections.get("self-assessment", "").strip()

    self_assessment = _parse_self_assessment(self_assessment_text)

    return IterationResult(
        plan=plan,
        thinking=thinking,
        memory=memory,
        enrichment=enrichment,
        self_assessment=self_assessment,
    )


def _extract_sections(text: str) -> dict[str, str]:
    """Extract named sections from markdown-style ``## Header`` blocks.

    Args:
        text: Markdown text with ``## Header`` sections.

    Returns:
        Dict mapping lowercase header names to their content.
    """
    sections: dict[str, str] = {}
    current_header: str | None = None
    current_lines: list[str] = []

    for line in text.split("\n"):
        header_match = re.match(r"^##\s+(.+)$", line)
        if header_match:
            if current_header is not None:
                sections[current_header] = "\n".join(current_lines)
            current_header = header_match.group(1).strip().lower()
            current_lines = []
        else:
            current_lines.append(line)

    if current_header is not None:
        sections[current_header] = "\n".join(current_lines)

    return sections


def _parse_self_assessment(text: str) -> SelfAssessment:
    """Parse a self-assessment section into a SelfAssessment.

    Expects lines like:
    - ``Quality: 8/10``
    - ``Gaps: some description``
    - ``Complete: yes`` or ``Complete: no``

    Args:
        text: Self-assessment section text.

    Returns:
        Parsed SelfAssessment.
    """
    quality = 0
    gaps = ""
    is_complete = False

    quality_match = re.search(r"Quality:\s*(\d+)/10", text)
    if quality_match:
        quality = int(quality_match.group(1))

    gaps_match = re.search(r"Gaps:\s*(.+)", text)
    if gaps_match:
        gaps = gaps_match.group(1).strip()

    complete_match = re.search(r"Complete:\s*(yes|no)", text, re.IGNORECASE)
    if complete_match:
        is_complete = complete_match.group(1).lower() == "yes"

    return SelfAssessment(quality=quality, gaps=gaps, is_complete=is_complete)


class EnrichmentLoopAgent:
    """Iterative enrichment agent using the Ralph Loop pattern.

    Runs multiple iterations of codebase exploration and enrichment
    refinement, accumulating state in external files. Each iteration
    the agent self-assesses quality and decides when to stop.

    Attributes:
        settings: Application settings with API keys and model config.
        scratch_path: Path to the scratch pad flow directory.
        max_iterations: Maximum number of refinement iterations.
    """

    def __init__(
        self,
        settings: NeevSettings,
        scratch_path: str,
        max_iterations: int = 3,
    ) -> None:
        """Initialize the enrichment loop agent.

        Args:
            settings: Application settings with anthropic_api_key.
            scratch_path: Path to the scratch pad flow directory
                for state file persistence.
            max_iterations: Maximum iterations before stopping (1-10).
        """
        self.settings = settings
        self.scratch_path = scratch_path
        self.max_iterations = max_iterations

    async def enrich(self, text: str, context: str | None = None) -> str:
        """Enrich text iteratively using the Ralph Loop pattern.

        Runs up to ``max_iterations`` iterations. Each iteration:
        1. Build an iteration-aware prompt with accumulated state
        2. Pipe prompt to the ``claude`` CLI subprocess
        3. Parse the structured response from stdout
        4. Update and persist state
        5. Check if the agent considers enrichment complete

        Uses the ``claude`` CLI command with ``--dangerously-skip-permissions``
        and ``--mcp-config`` flags. Iterations after the first use
        ``--continue`` to maintain conversation context.

        Args:
            text: Raw transcribed user speech to enrich.
            context: Optional additional context to prepend.

        Returns:
            The final enriched output as markdown.

        """
        flow_dir = Path(self.scratch_path)
        state = LoopState()

        transcript = text
        if context:
            transcript = f"{context}\n\n{text}"

        mcp_config = str(Path.home() / ".config" / "mcphub" / "servers.json")

        for iteration in range(1, self.max_iterations + 1):
            logger.info(
                "enrichment_loop_iteration",
                iteration=iteration,
                max_iterations=self.max_iterations,
            )

            prompt = build_iteration_prompt(transcript, state, iteration, self.max_iterations)
            full_prompt = f"{LOOP_SYSTEM_PROMPT}\n\n{prompt}"

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
                self.settings.claude_model,
                "--output-format",
                "text",
                "--mcp-config",
                mcp_config,
            ]
            if iteration > 1:
                cmd.append("--continue")

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await process.communicate(input=full_prompt.encode("utf-8"))

            if process.returncode != 0:
                logger.warning(
                    "claude_cli_error",
                    returncode=process.returncode,
                    stderr=stderr.decode("utf-8", errors="replace"),
                )

            raw_response = stdout.decode("utf-8")
            result = parse_structured_response(raw_response)
            state = state.update_from_result(result, iteration)
            write_state_files(state, flow_dir)

            logger.info(
                "enrichment_loop_iteration_done",
                iteration=iteration,
                quality=result.self_assessment.quality,
                is_complete=state.is_complete,
            )

            if state.is_complete:
                break

        logger.info(
            "enrichment_loop_done",
            total_iterations=state.iteration,
            draft_length=len(state.draft),
        )
        return state.draft
