"""Intent extraction module using Claude LLM.

Classifies user speech into intent categories (problem statement,
solution, clue, agreement, disagreement, etc.) and extracts
structured summaries using the EnrichmentAgent (Claude Agent SDK).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import structlog

from neev_voice.exceptions import NeevLLMError

logger = structlog.get_logger(__name__)

__all__ = [
    "DISCUSSION_INTENT_PROMPT",
    "INTENT_EXTRACTION_PROMPT",
    "EnrichmentAgentProtocol",
    "ExtractedIntent",
    "IntentCategory",
    "IntentExtractor",
]


class IntentCategory(StrEnum):
    """Categories of user intent extracted from speech.

    Attributes:
        PROBLEM_STATEMENT: User is describing a problem.
        SOLUTION: User is proposing a solution.
        CLUE: User is providing a hint or clue.
        MIXED: User speech contains multiple intent types.
        AGREEMENT: User agrees with a statement or proposal.
        DISAGREEMENT: User disagrees with a statement or proposal.
        QUESTION: User is asking a question.
    """

    PROBLEM_STATEMENT = "problem_statement"
    SOLUTION = "solution"
    CLUE = "clue"
    MIXED = "mixed"
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    QUESTION = "question"


@dataclass
class ExtractedIntent:
    """Structured result from intent extraction.

    Attributes:
        category: The classified intent category.
        summary: A concise summary of the user's speech.
        key_points: List of key points extracted from the speech.
        raw_text: The original transcribed text.
    """

    category: IntentCategory
    summary: str
    key_points: list[str]
    raw_text: str


INTENT_EXTRACTION_PROMPT = """Analyze the following transcribed speech \
(may be Hindi-English mixed) and extract the intent.

Respond ONLY with a valid JSON object (no markdown, no code fences) with these fields:
- "category": one of "problem_statement", "solution", "clue", "mixed", \
"agreement", "disagreement", "question"
- "summary": a concise 1-2 sentence summary
- "key_points": array of key points (strings)

Speech text:
{text}

JSON response:"""

DISCUSSION_INTENT_PROMPT = """Analyze the following response in a document \
discussion context (may be Hindi-English mixed).
Determine if the user agrees or disagrees with the content being discussed.

Look for agreement indicators: "haan", "theek hai", "sahi hai", "yes", \
"agreed", "okay", "bilkul", "correct"
Look for disagreement indicators: "nahi", "galat", "no", "disagree", \
"wrong", "nahi yaar", "I don't think so"

Respond ONLY with a valid JSON object (no markdown, no code fences) with these fields:
- "category": one of "agreement", "disagreement", "question", "mixed"
- "summary": a concise summary of their response
- "key_points": array of key points

User response:
{text}

Section being discussed:
{section}

JSON response:"""


class EnrichmentAgentProtocol(Protocol):
    """Protocol for enrichment agents compatible with IntentExtractor.

    Both EnrichmentAgent (v1) and EnrichmentLoopAgent (v2) satisfy
    this protocol by providing an async ``enrich`` method.
    """

    async def enrich(self, text: str, context: str | None = None) -> str:
        """Enrich text via LLM analysis.

        Args:
            text: Text to enrich.
            context: Optional additional context.

        Returns:
            Enriched output as a string.
        """
        ...


class IntentExtractor:
    """Extracts and classifies intent from transcribed speech.

    Uses an enrichment agent to analyze transcribed text and classify
    it into intent categories with structured summaries.

    Attributes:
        agent: An enrichment agent instance for LLM queries.
    """

    def __init__(self, agent: EnrichmentAgentProtocol) -> None:
        """Initialize IntentExtractor with an enrichment agent.

        Args:
            agent: Enrichment agent instance (v1 or v2) for LLM-powered analysis.
        """
        self.agent = agent

    async def extract(self, text: str) -> ExtractedIntent:
        """Extract intent from transcribed text.

        Sends the text to Claude for classification and structured
        extraction of intent, summary, and key points.

        Args:
            text: Transcribed speech text to analyze.

        Returns:
            ExtractedIntent with category, summary, and key points.

        Raises:
            NeevLLMError: If Claude query or response parsing fails.
        """
        logger.info("intent_extract_started", text_length=len(text))
        prompt = INTENT_EXTRACTION_PROMPT.format(text=text)
        response = await self.agent.enrich(prompt)
        result = self._parse_intent_response(response, text)
        logger.info("intent_extract_done", category=result.category.value)
        return result

    async def extract_discussion_intent(self, text: str, section: str) -> ExtractedIntent:
        """Extract intent in a document discussion context.

        Analyzes user response to determine agreement/disagreement
        with the section being discussed.

        Args:
            text: User's transcribed response.
            section: The document section being discussed.

        Returns:
            ExtractedIntent with discussion-specific classification.

        Raises:
            NeevLLMError: If Claude query or response parsing fails.
        """
        logger.info("discussion_intent_started", text_length=len(text))
        prompt = DISCUSSION_INTENT_PROMPT.format(text=text, section=section)
        response = await self.agent.enrich(prompt)
        result = self._parse_intent_response(response, text)
        logger.info("discussion_intent_done", category=result.category.value)
        return result

    @staticmethod
    def _parse_intent_response(response: str, raw_text: str) -> ExtractedIntent:
        """Parse Claude's JSON response into an ExtractedIntent.

        Handles both clean JSON and responses wrapped in markdown
        code fences.

        Args:
            response: Raw response string from Claude.
            raw_text: Original transcribed text for reference.

        Returns:
            Parsed ExtractedIntent.

        Raises:
            NeevLLMError: If the response cannot be parsed as valid JSON.
        """
        # Strip markdown code fences if present
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last fence lines
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
