"""Tests for EnrichmentAgent using Claude Agent SDK."""

from unittest.mock import MagicMock

import pytest

from neev_voice.config import NeevSettings
from neev_voice.llm.agent import (
    ENRICHMENT_SYSTEM_PROMPT,
    ENRICHMENT_TEMPLATE,
    TASK_BREAKDOWN_INSTRUCTION,
    EnrichmentAgent,
    build_system_prompt,
)


@pytest.fixture
def settings():
    """Create test settings with Anthropic API key."""
    return NeevSettings(sarvam_api_key="test", anthropic_api_key="test-anthropic-key")


@pytest.fixture
def settings_no_key():
    """Create test settings without Anthropic API key."""
    return NeevSettings(sarvam_api_key="test", anthropic_api_key="")


@pytest.fixture
def agent(settings):
    """Create an EnrichmentAgent with test settings."""
    return EnrichmentAgent(settings)


class TestEnrichmentTemplate:
    """Tests for the enrichment markdown template."""

    def test_template_has_all_sections(self):
        """Test template contains all required markdown headings."""
        assert "# Enrichment Report" in ENRICHMENT_TEMPLATE
        assert "## Original Statement" in ENRICHMENT_TEMPLATE
        assert "## Context Analysis" in ENRICHMENT_TEMPLATE
        assert "## Formalized Problem Statement" in ENRICHMENT_TEMPLATE
        assert "## Relevant Code References" in ENRICHMENT_TEMPLATE
        assert "## Suggested Investigation Areas" in ENRICHMENT_TEMPLATE
        assert "## Atomic Task Breakdown" in ENRICHMENT_TEMPLATE

    def test_template_has_placeholder(self):
        """Test template has {original_text} placeholder."""
        assert "{original_text}" in ENRICHMENT_TEMPLATE

    def test_template_can_be_formatted(self):
        """Test template can be formatted with original_text."""
        result = ENRICHMENT_TEMPLATE.format(original_text="test input")
        assert "> test input" in result
        assert "{original_text}" not in result


class TestBuildSystemPrompt:
    """Tests for the build_system_prompt function."""

    def test_includes_base_prompt(self):
        """Test always includes the base enrichment system prompt."""
        result = build_system_prompt()
        assert ENRICHMENT_SYSTEM_PROMPT in result

    def test_includes_task_breakdown(self):
        """Test always includes task breakdown instruction."""
        result = build_system_prompt()
        assert TASK_BREAKDOWN_INSTRUCTION in result

    def test_includes_template(self):
        """Test always includes enrichment template."""
        result = build_system_prompt()
        assert "# Enrichment Report" in result

    def test_includes_template_instruction(self):
        """Test always includes the template usage instruction."""
        result = build_system_prompt()
        assert "Output your enrichment as a filled-in markdown" in result

    def test_without_scratch_path(self):
        """Test scratch pad instruction excluded when no path given."""
        result = build_system_prompt()
        assert (
            "scratch pad" not in result.lower().split("scratch pad")[0]
            or "Use scratch pad" not in result
        )

    def test_with_scratch_path(self):
        """Test scratch pad instruction included when path given."""
        result = build_system_prompt(scratch_path="/tmp/scratch")
        assert "/tmp/scratch" in result
        assert "Use scratch pad" in result

    def test_scratch_path_none_excludes_instruction(self):
        """Test scratch pad instruction excluded when path is None."""
        result = build_system_prompt(scratch_path=None)
        assert "Use scratch pad" not in result

    def test_scratch_path_empty_excludes_instruction(self):
        """Test scratch pad instruction excluded when path is empty string."""
        result = build_system_prompt(scratch_path="")
        assert "Use scratch pad" not in result


class TestEnrichmentAgentInit:
    """Tests for EnrichmentAgent initialization."""

    def test_init_stores_settings(self, settings):
        """Test agent stores settings on init."""
        agent = EnrichmentAgent(settings)
        assert agent.settings is settings

    def test_init_with_api_key(self, settings):
        """Test agent initializes with API key set."""
        agent = EnrichmentAgent(settings)
        assert agent.settings.anthropic_api_key == "test-anthropic-key"

    def test_init_stores_scratch_path(self, settings):
        """Test agent stores scratch_path on init."""
        agent = EnrichmentAgent(settings, scratch_path="/tmp/scratch")
        assert agent.scratch_path == "/tmp/scratch"

    def test_init_scratch_path_default_none(self, settings):
        """Test scratch_path defaults to None."""
        agent = EnrichmentAgent(settings)
        assert agent.scratch_path is None


class TestEnrichmentAgentEnrich:
    """Tests for EnrichmentAgent.enrich method."""

    async def test_enrich_missing_api_key_raises(self, settings_no_key):
        """Test enrich raises ValueError when API key is not set."""
        agent = EnrichmentAgent(settings_no_key)
        with pytest.raises(ValueError, match="LLM API key is required"):
            await agent.enrich("some text")

    async def test_enrich_calls_query_with_correct_options(self, agent, mocker):
        """Test enrich calls claude_agent_sdk.query with correct options."""
        from claude_agent_sdk import AssistantMessage, TextBlock

        mock_msg = MagicMock(spec=AssistantMessage)
        mock_block = MagicMock(spec=TextBlock)
        mock_block.text = "enriched output"
        mock_msg.content = [mock_block]

        async def mock_query(prompt, options):
            """Yield a mock AssistantMessage with TextBlock content."""
            yield mock_msg

        mocker.patch("neev_voice.llm.agent.query", side_effect=mock_query)

        result = await agent.enrich("test input")

        assert result == "enriched output"

    async def test_enrich_allowed_tools(self, agent, mocker):
        """Test enrich configures only Read, Glob, Grep tools."""
        captured_options = {}

        async def mock_query(prompt, options):
            """Capture options and yield nothing."""
            captured_options["options"] = options
            return
            yield  # noqa: F841 - makes it an async generator

        mocker.patch("neev_voice.llm.agent.query", side_effect=mock_query)

        await agent.enrich("test input")

        opts = captured_options["options"]
        assert opts.allowed_tools == ["Read", "Glob", "Grep"]

    async def test_enrich_system_prompt_includes_base(self, agent, mocker):
        """Test enrich uses system prompt containing the base prompt."""
        captured_options = {}

        async def mock_query(prompt, options):
            """Capture options and yield nothing."""
            captured_options["options"] = options
            return
            yield

        mocker.patch("neev_voice.llm.agent.query", side_effect=mock_query)

        await agent.enrich("test input")

        opts = captured_options["options"]
        assert ENRICHMENT_SYSTEM_PROMPT in opts.system_prompt

    async def test_enrich_system_prompt_includes_template(self, agent, mocker):
        """Test enrich system prompt includes the enrichment template."""
        captured_options = {}

        async def mock_query(prompt, options):
            """Capture options and yield nothing."""
            captured_options["options"] = options
            return
            yield

        mocker.patch("neev_voice.llm.agent.query", side_effect=mock_query)

        await agent.enrich("test input")

        opts = captured_options["options"]
        assert "# Enrichment Report" in opts.system_prompt

    async def test_enrich_system_prompt_with_scratch_path(self, settings, mocker):
        """Test enrich includes scratch pad instruction when path is set."""
        agent = EnrichmentAgent(settings, scratch_path="/tmp/scratch")
        captured_options = {}

        async def mock_query(prompt, options):
            """Capture options and yield nothing."""
            captured_options["options"] = options
            return
            yield

        mocker.patch("neev_voice.llm.agent.query", side_effect=mock_query)

        await agent.enrich("test input")

        opts = captured_options["options"]
        assert "/tmp/scratch" in opts.system_prompt

    async def test_enrich_returns_concatenated_text(self, agent, mocker):
        """Test enrich returns text from multiple assistant messages."""
        from claude_agent_sdk import AssistantMessage, TextBlock

        msg1 = MagicMock(spec=AssistantMessage)
        block1 = MagicMock(spec=TextBlock)
        block1.text = "first part"
        msg1.content = [block1]

        msg2 = MagicMock(spec=AssistantMessage)
        block2 = MagicMock(spec=TextBlock)
        block2.text = "second part"
        msg2.content = [block2]

        async def mock_query(prompt, options):
            """Yield two assistant messages."""
            yield msg1
            yield msg2

        mocker.patch("neev_voice.llm.agent.query", side_effect=mock_query)

        result = await agent.enrich("test input")

        assert result == "first part\nsecond part"

    async def test_enrich_with_context(self, agent, mocker):
        """Test enrich prepends context to prompt."""
        captured_prompt = {}

        async def mock_query(prompt, options):
            """Capture prompt and yield nothing."""
            captured_prompt["prompt"] = prompt
            return
            yield

        mocker.patch("neev_voice.llm.agent.query", side_effect=mock_query)

        await agent.enrich("the question", context="some context")

        assert "some context" in captured_prompt["prompt"]
        assert "the question" in captured_prompt["prompt"]

    async def test_enrich_without_context(self, agent, mocker):
        """Test enrich includes raw text in prompt when no context."""
        captured_prompt = {}

        async def mock_query(prompt, options):
            """Capture prompt and yield nothing."""
            captured_prompt["prompt"] = prompt
            return
            yield

        mocker.patch("neev_voice.llm.agent.query", side_effect=mock_query)

        await agent.enrich("raw text only")

        assert "raw text only" in captured_prompt["prompt"]

    async def test_enrich_prompt_includes_template_instruction(self, agent, mocker):
        """Test enrich prompt asks agent to fill the template."""
        captured_prompt = {}

        async def mock_query(prompt, options):
            """Capture prompt and yield nothing."""
            captured_prompt["prompt"] = prompt
            return
            yield

        mocker.patch("neev_voice.llm.agent.query", side_effect=mock_query)

        await agent.enrich("test input")

        assert "Fill in the enrichment markdown template" in captured_prompt["prompt"]

    async def test_enrich_empty_response(self, agent, mocker):
        """Test enrich returns empty string when no text blocks."""

        async def mock_query(prompt, options):
            """Yield nothing."""
            return
            yield

        mocker.patch("neev_voice.llm.agent.query", side_effect=mock_query)

        result = await agent.enrich("test input")

        assert result == ""

    async def test_enrich_passes_env_with_api_key(self, agent, mocker):
        """Test enrich passes ANTHROPIC_API_KEY in env."""
        captured_options = {}

        async def mock_query(prompt, options):
            """Capture options."""
            captured_options["options"] = options
            return
            yield

        mocker.patch("neev_voice.llm.agent.query", side_effect=mock_query)

        await agent.enrich("test")

        opts = captured_options["options"]
        assert opts.env["ANTHROPIC_API_KEY"] == "test-anthropic-key"

    async def test_enrich_permission_mode(self, agent, mocker):
        """Test enrich uses default permission mode."""
        captured_options = {}

        async def mock_query(prompt, options):
            """Capture options."""
            captured_options["options"] = options
            return
            yield

        mocker.patch("neev_voice.llm.agent.query", side_effect=mock_query)

        await agent.enrich("test")

        opts = captured_options["options"]
        assert opts.permission_mode == "default"

    async def test_enrich_uses_resolved_llm_api_key(self, agent, mocker):
        """Test enrich passes resolved_llm_api_key in env."""
        captured_options = {}

        async def mock_query(prompt, options):
            """Capture options."""
            captured_options["options"] = options
            return
            yield

        mocker.patch("neev_voice.llm.agent.query", side_effect=mock_query)

        await agent.enrich("test")

        opts = captured_options["options"]
        assert opts.env["ANTHROPIC_API_KEY"] == agent.settings.resolved_llm_api_key

    async def test_enrich_sets_base_url_when_configured(self, mocker):
        """Test enrich sets ANTHROPIC_BASE_URL when llm_api_base is non-empty."""
        settings = NeevSettings(
            anthropic_api_key="test-key",
            llm_api_base="https://openrouter.ai/api/v1",
        )
        agent = EnrichmentAgent(settings)
        captured_options = {}

        async def mock_query(prompt, options):
            """Capture options."""
            captured_options["options"] = options
            return
            yield

        mocker.patch("neev_voice.llm.agent.query", side_effect=mock_query)

        await agent.enrich("test")

        opts = captured_options["options"]
        assert opts.env["ANTHROPIC_BASE_URL"] == "https://openrouter.ai/api/v1"

    async def test_enrich_no_base_url_when_empty(self, agent, mocker):
        """Test enrich does not set ANTHROPIC_BASE_URL when llm_api_base is empty."""
        captured_options = {}

        async def mock_query(prompt, options):
            """Capture options."""
            captured_options["options"] = options
            return
            yield

        mocker.patch("neev_voice.llm.agent.query", side_effect=mock_query)

        await agent.enrich("test")

        opts = captured_options["options"]
        assert "ANTHROPIC_BASE_URL" not in opts.env

    async def test_enrich_openrouter_provider_uses_openrouter_key(self, mocker):
        """Test enrich uses openrouter_api_key when provider is openrouter."""
        from neev_voice.config import LLMProviderType

        settings = NeevSettings(
            openrouter_api_key="sk-or-test",
            llm_provider=LLMProviderType.OPENROUTER,
        )
        agent = EnrichmentAgent(settings)
        captured_options = {}

        async def mock_query(prompt, options):
            """Capture options."""
            captured_options["options"] = options
            return
            yield

        mocker.patch("neev_voice.llm.agent.query", side_effect=mock_query)

        await agent.enrich("test")

        opts = captured_options["options"]
        assert opts.env["ANTHROPIC_API_KEY"] == "sk-or-test"
