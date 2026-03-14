"""Prepare-enquiry engine for the discuss state machine.

Researches answers to user enquiries using Claude CLI and generates
TTS-ready transcripts for the presentation-enquiry state.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from neev_voice.discuss.session import SessionInfo

if TYPE_CHECKING:
    from neev_voice.config import NeevSettings

logger = structlog.get_logger(__name__)

__all__ = ["PrepareEnquiryEngine"]


ENQUIRY_RESEARCH_PROMPT = """\
You are a research assistant answering a user's question about a codebase
and its associated research documents.

The user is currently learning about a system through an interactive
presentation. They have paused to ask a question.

Research documents are located at: {research_path}
Source code is at: {source_path}

User's question:
---
{query}
---

Instructions:
1. Research the answer using the available tools (Read, Glob, Grep)
2. Provide a clear, concise answer focused on the question
3. Reference specific files and code when relevant
4. Structure your answer in two sections:

## Answer
[Clear answer to the question, with code references]

## Transcript
[TTS-ready version of the answer — natural spoken language,
no markdown formatting, spell out abbreviations, suitable for
text-to-speech synthesis]
"""

ENQUIRY_UPDATE_PROMPT = """\
You are a research assistant updating a previous answer based on a
follow-up question from the user.

The user previously asked a question and received an answer. They have
now asked a follow-up question that requires the previous answer to
be updated or expanded.

Research documents are located at: {research_path}
Source code is at: {source_path}

Previous answer:
---
{previous_answer}
---

Follow-up question:
---
{query}
---

Instructions:
1. Review the previous answer
2. Research additional information for the follow-up question
3. Update the answer to incorporate the new information
4. Start the transcript from the correction/addition point

## Answer
[Updated answer incorporating the follow-up]

## Transcript
[TTS-ready version starting from the correction point]
"""


class PrepareEnquiryEngine:
    """Researches answers to user enquiries using Claude CLI.

    Handles two scenarios:
    - New enquiry (from presentation state): generates fresh answer
    - Follow-up enquiry (from presentation-enquiry state): updates
      previous answer with new context

    Attributes:
        session: Current discuss session info.
        settings: Application settings.
        session_dir: Path to the session's scratch pad directory.
    """

    def __init__(
        self,
        session: SessionInfo,
        settings: NeevSettings,
        session_dir: Path | None = None,
    ) -> None:
        """Initialize the prepare-enquiry engine.

        Args:
            session: Current discuss session info.
            settings: Application settings.
            session_dir: Override session directory path.
        """
        self.session = session
        self.settings = settings
        self.session_dir = session_dir or Path(settings.discuss_base_dir) / session.name

    async def run(
        self,
        query: str,
        from_presentation_enquiry: bool = False,
        previous_answer: str | None = None,
    ) -> str:
        """Research an answer to the user's enquiry.

        Args:
            query: The user's question text.
            from_presentation_enquiry: True if this is a follow-up
                from a presentation-enquiry → enquiry path.
            previous_answer: Previous answer to update (for follow-ups).

        Returns:
            TTS-ready transcript of the answer.
        """
        logger.info(
            "prepare_enquiry_started",
            query_length=len(query),
            is_followup=from_presentation_enquiry,
        )

        if from_presentation_enquiry and previous_answer:
            prompt = ENQUIRY_UPDATE_PROMPT.format(
                research_path=self.session.research_path,
                source_path=self.session.source_path,
                previous_answer=previous_answer,
                query=query,
            )
        else:
            prompt = ENQUIRY_RESEARCH_PROMPT.format(
                research_path=self.session.research_path,
                source_path=self.session.source_path,
                query=query,
            )

        response = await self._run_claude(prompt)

        # Parse response for answer and transcript
        answer, transcript = self._parse_response(response)

        # Save enquiry results
        self._save_enquiry(query, answer, transcript)

        logger.info(
            "prepare_enquiry_complete",
            answer_length=len(answer),
            transcript_length=len(transcript),
        )

        return transcript

    def _parse_response(self, response: str) -> tuple[str, str]:
        """Parse Claude's response into answer and transcript sections.

        Args:
            response: Raw Claude CLI output.

        Returns:
            Tuple of (answer text, TTS-ready transcript).
        """
        import re

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

        answer = sections.get("answer", response.strip())
        transcript = sections.get("transcript", answer)

        return answer, transcript

    def _save_enquiry(self, query: str, answer: str, transcript: str) -> None:
        """Save enquiry results to the session directory.

        Args:
            query: The user's question.
            answer: The researched answer.
            transcript: TTS-ready transcript of the answer.
        """
        timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        enquiry_dir = self.session_dir / "enquiries" / timestamp
        enquiry_dir.mkdir(parents=True, exist_ok=True)

        (enquiry_dir / "query.txt").write_text(query, encoding="utf-8")
        (enquiry_dir / "answer.md").write_text(answer, encoding="utf-8")
        (enquiry_dir / "transcript.md").write_text(transcript, encoding="utf-8")

        # Also save metadata
        metadata = {
            "timestamp": datetime.now(UTC).isoformat(),
            "query_length": len(query),
            "answer_length": len(answer),
            "transcript_length": len(transcript),
        }
        (enquiry_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

    async def _run_claude(self, prompt: str) -> str:
        """Run a prompt through Claude CLI.

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

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            logger.warning(
                "prepare_enquiry_claude_error",
                returncode=process.returncode,
                stderr=stderr.decode("utf-8", errors="replace")[:500],
            )

        return stdout.decode("utf-8")
