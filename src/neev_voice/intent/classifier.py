"""Lightweight intent classifier using Claude CLI subprocess.

Provides single-turn intent classification without codebase exploration.
Uses the ``claude`` CLI command to classify user speech into intent
categories and extract structured JSON responses. This is separate from
enrichment — the classifier only categorizes intent, while enrichment
agents produce detailed markdown reports.
"""

from __future__ import annotations

import asyncio
import json
import os

import structlog

from neev_voice.config import NeevSettings
from neev_voice.exceptions import NeevLLMError
from neev_voice.intent.extractor import (
    DISCUSSION_INTENT_PROMPT,
    INTENT_EXTRACTION_PROMPT,
    ExtractedIntent,
    IntentCategory,
)

logger = structlog.get_logger(__name__)

__all__ = ["IntentClassifier"]


class IntentClassifier:
    """Lightweight intent classifier using Claude CLI subprocess.

    Classifies user speech into intent categories using a single-turn
    Claude CLI call. Does not explore the codebase — only performs
    intent classification and returns structured ``ExtractedIntent``.

    Attributes:
        settings: Application settings with model config.
    """

    def __init__(self, settings: NeevSettings) -> None:
        """Initialize IntentClassifier with application settings.

        Args:
            settings: Application settings with claude_model configuration.
        """
        self.settings = settings

    async def classify(self, text: str) -> ExtractedIntent:
        """Classify transcribed speech into an intent category.

        Sends the text to the Claude CLI for single-turn intent
        classification and parses the JSON response.

        Args:
            text: Transcribed speech text to classify.

        Returns:
            ExtractedIntent with category, summary, and key points.

        Raises:
            NeevLLMError: If the CLI call fails or response parsing fails.
        """
        prompt = INTENT_EXTRACTION_PROMPT.format(text=text)
        response = await self._run_claude(prompt)
        return self._parse_response(response, text)

    async def classify_discussion(self, text: str, section: str) -> ExtractedIntent:
        """Classify user response in a document discussion context.

        Analyzes user speech to determine agreement/disagreement with
        a document section being discussed.

        Args:
            text: User's transcribed response.
            section: The document section being discussed.

        Returns:
            ExtractedIntent with discussion-specific classification.

        Raises:
            NeevLLMError: If the CLI call fails or response parsing fails.
        """
        prompt = DISCUSSION_INTENT_PROMPT.format(text=text, section=section)
        response = await self._run_claude(prompt)
        return self._parse_response(response, text)

    async def _run_claude(self, prompt: str) -> str:
        """Run the Claude CLI subprocess with the given prompt.

        Invokes the ``claude`` CLI command with ``--dangerously-skip-permissions``
        and pipes the prompt via stdin.

        Args:
            prompt: The full prompt to send to Claude.

        Returns:
            Raw stdout text from the subprocess.

        Raises:
            NeevLLMError: If the CLI returns non-zero exit code with no output.
        """
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
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await process.communicate(input=prompt.encode("utf-8"))

        raw_output = stdout.decode("utf-8")

        if process.returncode != 0 and not raw_output.strip():
            logger.warning(
                "claude_cli_failed",
                returncode=process.returncode,
                stderr=stderr.decode("utf-8", errors="replace"),
            )
            raise NeevLLMError(
                f"Claude CLI intent classification failed with exit code {process.returncode}"
            )

        return raw_output

    @staticmethod
    def _parse_response(response: str, raw_text: str) -> ExtractedIntent:
        """Parse Claude's JSON response into an ExtractedIntent.

        Handles both clean JSON and responses wrapped in markdown
        code fences (```json ... ```).

        Args:
            response: Raw response string from Claude.
            raw_text: Original transcribed text for reference.

        Returns:
            Parsed ExtractedIntent.

        Raises:
            NeevLLMError: If the response cannot be parsed as valid JSON.
        """
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise NeevLLMError(f"Failed to parse intent response: {e}\nRaw: {response}") from e

        category_str = data.get("category", "mixed")
        try:
            category = IntentCategory(category_str)
        except ValueError:
            category = IntentCategory.MIXED

        return ExtractedIntent(
            category=category,
            summary=data.get("summary", ""),
            key_points=data.get("key_points", []),
            raw_text=raw_text,
        )
