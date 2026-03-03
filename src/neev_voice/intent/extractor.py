"""Intent extraction module using Claude LLM.

Classifies user speech into intent categories (problem statement,
solution, clue, agreement, disagreement, etc.) and extracts
structured summaries using the EnrichmentAgent (Claude Agent SDK).
"""

import json
from dataclasses import dataclass
from enum import Enum

from neev_voice.llm.agent import EnrichmentAgent


class IntentCategory(str, Enum):
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


class IntentExtractor:
    """Extracts and classifies intent from transcribed speech.

    Uses the EnrichmentAgent (Claude Agent SDK) to analyze transcribed
    text and classify it into intent categories with structured summaries.

    Attributes:
        agent: The EnrichmentAgent instance for LLM queries.
    """

    def __init__(self, agent: EnrichmentAgent) -> None:
        """Initialize IntentExtractor with an EnrichmentAgent.

        Args:
            agent: EnrichmentAgent instance for LLM-powered analysis.
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
            RuntimeError: If Claude query or response parsing fails.
        """
        prompt = INTENT_EXTRACTION_PROMPT.format(text=text)
        response = await self.agent.enrich(prompt)
        return self._parse_intent_response(response, text)

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
            RuntimeError: If Claude query or response parsing fails.
        """
        prompt = DISCUSSION_INTENT_PROMPT.format(text=text, section=section)
        response = await self.agent.enrich(prompt)
        return self._parse_intent_response(response, text)

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
            RuntimeError: If the response cannot be parsed as valid JSON.
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
            raise RuntimeError(f"Failed to parse intent response: {e}\nRaw: {response}")

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
