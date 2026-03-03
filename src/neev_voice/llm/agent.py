"""Enrichment agent using Claude Agent SDK.

Replaces the broken ClaudeCodeClient subprocess approach with the
Claude Agent SDK.  The agent explores the codebase with read-only
tools (Read, Glob, Grep) and enriches/formalizes the user's problem
statement.  It does **not** write or modify any code.
"""

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    TextBlock,
    query,
)

from neev_voice.config import NeevSettings

ENRICHMENT_SYSTEM_PROMPT = (
    "You are a code-aware analyst. Given user speech (possibly Hindi-English), "
    "explore the codebase to understand context, then produce a formal, specific "
    "enrichment of the user's problem statement. Reference actual files and "
    "functions where relevant. Do NOT suggest or make code changes."
)

ENRICHMENT_TEMPLATE = """\
# Enrichment Report

## Original Statement
> {original_text}

## Context Analysis
<!-- Describe relevant codebase context discovered via Read/Glob/Grep -->

## Formalized Problem Statement
<!-- Rewrite the original speech as a precise, actionable problem statement -->

## Relevant Code References
<!-- List actual files/functions/classes that relate to this problem -->
| File | Symbol | Relevance |
|------|--------|-----------|

## Suggested Investigation Areas
<!-- Areas to explore further, without making code changes -->
-

## Atomic Task Breakdown
<!-- Break the problem into logical, atomic tasks -->
1.
"""

SCRATCH_PAD_INSTRUCTION = (
    "Use scratch pad `{scratch_path}` for memory for ephemeral storage, thinking, memory, plan etc"
)

TASK_BREAKDOWN_INSTRUCTION = (
    "Breakdown the tasks into logical and atomic tasks. Think Step by step "
    "before creating atomic tasks. Proceed to one task at a time."
)

TEMPLATE_INSTRUCTION = (
    "Output your enrichment as a filled-in markdown using the template below. "
    "Replace the HTML comments with your findings. Keep all headings.\n\n"
)


def build_system_prompt(scratch_path: str | None = None) -> str:
    """Build the full system prompt for the enrichment agent.

    Combines the base enrichment prompt with optional scratch pad
    instruction, task breakdown instruction, and template instruction.

    Args:
        scratch_path: Optional path to the scratch pad directory.
            When provided, the scratch pad instruction is included.

    Returns:
        The assembled system prompt string.
    """
    parts = [ENRICHMENT_SYSTEM_PROMPT]

    if scratch_path:
        parts.append(SCRATCH_PAD_INSTRUCTION.format(scratch_path=scratch_path))

    parts.append(TASK_BREAKDOWN_INSTRUCTION)
    parts.append(TEMPLATE_INSTRUCTION + ENRICHMENT_TEMPLATE)

    return "\n\n".join(parts)


class EnrichmentAgent:
    """Enriches user problem statements by exploring codebase context.

    Uses Claude Agent SDK with read-only tools (Read, Glob, Grep)
    to analyze the codebase and produce formal, specific enrichments
    of user speech.  Does not modify any code.

    Attributes:
        settings: Application settings containing Anthropic API key and model config.
        scratch_path: Optional path to scratch pad directory for agent memory.
    """

    def __init__(self, settings: NeevSettings, scratch_path: str | None = None) -> None:
        """Initialize EnrichmentAgent with application settings.

        Args:
            settings: Application settings with anthropic_api_key.
            scratch_path: Optional scratch pad directory path.
        """
        self.settings = settings
        self.scratch_path = scratch_path

    async def enrich(self, text: str, context: str | None = None) -> str:
        """Enrich a user's raw text into a formal problem statement.

        Sends the text to a Claude agent configured with read-only
        codebase tools and collects the enriched output. The agent
        fills in the enrichment markdown template.

        Args:
            text: Raw transcribed user speech.
            context: Optional additional context to prepend.

        Returns:
            The enriched, formalized problem statement as markdown.

        Raises:
            ValueError: If the Anthropic API key is not configured.
        """
        if not self.settings.resolved_llm_api_key:
            raise ValueError(
                "LLM API key is required for enrichment. "
                "Set NEEV_ANTHROPIC_API_KEY or NEEV_OPENROUTER_API_KEY."
            )

        prompt_parts = []
        if context:
            prompt_parts.append(context)
        prompt_parts.append(text)
        prompt_parts.append(
            "Fill in the enrichment markdown template based on your "
            "codebase analysis. Replace the HTML comments with your "
            "findings. Keep all headings."
        )
        prompt = "\n\n".join(prompt_parts)

        system_prompt = build_system_prompt(self.scratch_path)

        env = {"ANTHROPIC_API_KEY": self.settings.resolved_llm_api_key}
        if self.settings.resolved_llm_api_base:
            env["ANTHROPIC_BASE_URL"] = self.settings.resolved_llm_api_base

        options = ClaudeAgentOptions(
            allowed_tools=["Read", "Glob", "Grep"],
            system_prompt=system_prompt,
            permission_mode="default",
            model=self.settings.claude_model,
            max_turns=5,
            env=env,
        )

        texts: list[str] = []
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        texts.append(block.text)

        return "\n".join(texts) if texts else ""
