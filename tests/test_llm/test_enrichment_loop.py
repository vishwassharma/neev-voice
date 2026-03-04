"""Tests for the enrichment loop state management module.

Tests LoopState dataclass, IterationResult parsing,
state file I/O operations, and EnrichmentLoopAgent
with claude CLI subprocess integration.
"""

import json
from unittest.mock import AsyncMock

import pytest

from neev_voice.config import NeevSettings
from neev_voice.llm.enrichment_loop import (
    LOOP_RESPONSE_FORMAT,
    LOOP_SYSTEM_PROMPT,
    EnrichmentLoopAgent,
    IterationResult,
    LoopState,
    SelfAssessment,
    build_iteration_prompt,
    parse_structured_response,
    read_state_files,
    write_state_files,
)


class TestSelfAssessment:
    """Tests for SelfAssessment dataclass."""

    def test_defaults(self):
        """Test SelfAssessment default values."""
        assessment = SelfAssessment()
        assert assessment.quality == 0
        assert assessment.gaps == ""
        assert assessment.is_complete is False

    def test_custom_values(self):
        """Test SelfAssessment with custom values."""
        assessment = SelfAssessment(quality=8, gaps="none", is_complete=True)
        assert assessment.quality == 8
        assert assessment.gaps == "none"
        assert assessment.is_complete is True


class TestIterationResult:
    """Tests for IterationResult dataclass."""

    def test_defaults(self):
        """Test IterationResult default values."""
        result = IterationResult()
        assert result.plan == ""
        assert result.thinking == ""
        assert result.memory == ""
        assert result.enrichment == ""
        assert isinstance(result.self_assessment, SelfAssessment)

    def test_custom_values(self):
        """Test IterationResult with custom values."""
        result = IterationResult(
            plan="My plan",
            thinking="My reasoning",
            memory="Key discovery",
            enrichment="Enriched output",
            self_assessment=SelfAssessment(quality=9, is_complete=True),
        )
        assert result.plan == "My plan"
        assert result.thinking == "My reasoning"
        assert result.memory == "Key discovery"
        assert result.enrichment == "Enriched output"
        assert result.self_assessment.quality == 9


class TestLoopState:
    """Tests for LoopState dataclass."""

    def test_initial_state(self):
        """Test LoopState with default initial values."""
        state = LoopState()
        assert state.iteration == 0
        assert state.plan == ""
        assert state.thinking == ""
        assert state.draft == ""
        assert state.memory == ""
        assert state.is_complete is False

    def test_custom_state(self):
        """Test LoopState with custom values."""
        state = LoopState(
            iteration=2,
            plan="Analysis plan",
            thinking="Current reasoning",
            draft="Draft enrichment",
            memory="Key findings",
            is_complete=True,
        )
        assert state.iteration == 2
        assert state.plan == "Analysis plan"
        assert state.is_complete is True

    def test_update_from_result_first_iteration(self):
        """Test updating state from first iteration result."""
        state = LoopState()
        result = IterationResult(
            plan="New plan",
            thinking="New thinking",
            memory="New memory",
            enrichment="New draft",
            self_assessment=SelfAssessment(quality=5, gaps="some gaps"),
        )
        updated = state.update_from_result(result, iteration=1)
        assert updated.iteration == 1
        assert updated.plan == "New plan"
        assert updated.thinking == "New thinking"
        assert updated.draft == "New draft"
        assert updated.memory == "New memory"
        assert updated.is_complete is False

    def test_update_from_result_appends_memory(self):
        """Test subsequent iterations append to memory."""
        state = LoopState(iteration=1, memory="First discovery")
        result = IterationResult(
            thinking="More thinking",
            memory="Second discovery",
            enrichment="Better draft",
            self_assessment=SelfAssessment(quality=7),
        )
        updated = state.update_from_result(result, iteration=2)
        assert "First discovery" in updated.memory
        assert "Second discovery" in updated.memory

    def test_update_from_result_preserves_plan(self):
        """Test subsequent iterations preserve existing plan."""
        state = LoopState(iteration=1, plan="Original plan")
        result = IterationResult(
            plan="",  # no plan on iteration 2
            thinking="More analysis",
            enrichment="Updated draft",
            self_assessment=SelfAssessment(quality=8),
        )
        updated = state.update_from_result(result, iteration=2)
        assert updated.plan == "Original plan"

    def test_update_from_result_complete(self):
        """Test state is marked complete when self-assessment says so."""
        state = LoopState(iteration=1)
        result = IterationResult(
            enrichment="Final draft",
            self_assessment=SelfAssessment(quality=9, is_complete=True),
        )
        updated = state.update_from_result(result, iteration=2)
        assert updated.is_complete is True


class TestWriteStateFiles:
    """Tests for write_state_files function."""

    def test_writes_plan(self, tmp_path):
        """Test plan.md is written."""
        state = LoopState(plan="My analysis plan")
        write_state_files(state, tmp_path)
        assert (tmp_path / "plan.md").read_text() == "My analysis plan"

    def test_writes_thinking(self, tmp_path):
        """Test thinking.md is written."""
        state = LoopState(thinking="Current reasoning")
        write_state_files(state, tmp_path)
        assert (tmp_path / "thinking.md").read_text() == "Current reasoning"

    def test_writes_enriched_draft(self, tmp_path):
        """Test enriched_draft.md is written."""
        state = LoopState(draft="Draft content")
        write_state_files(state, tmp_path)
        assert (tmp_path / "enriched_draft.md").read_text() == "Draft content"

    def test_writes_memory(self, tmp_path):
        """Test memory.md is written."""
        state = LoopState(memory="Key findings")
        write_state_files(state, tmp_path)
        assert (tmp_path / "memory.md").read_text() == "Key findings"

    def test_appends_to_loop_log(self, tmp_path):
        """Test loop_log.jsonl is appended with iteration data."""
        state = LoopState(iteration=1, draft="draft", thinking="think")
        write_state_files(state, tmp_path)
        log_path = tmp_path / "loop_log.jsonl"
        assert log_path.exists()
        entry = json.loads(log_path.read_text().strip())
        assert entry["iteration"] == 1
        assert entry["is_complete"] is False

    def test_loop_log_appends_multiple(self, tmp_path):
        """Test loop_log.jsonl accumulates entries."""
        state1 = LoopState(iteration=1)
        state2 = LoopState(iteration=2)
        write_state_files(state1, tmp_path)
        write_state_files(state2, tmp_path)
        lines = (tmp_path / "loop_log.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

    def test_creates_directory(self, tmp_path):
        """Test creates flow_dir if it doesn't exist."""
        flow_dir = tmp_path / "nested" / "dir"
        state = LoopState(plan="test")
        write_state_files(state, flow_dir)
        assert (flow_dir / "plan.md").exists()

    def test_empty_fields_still_write(self, tmp_path):
        """Test empty state fields still create files."""
        state = LoopState()
        write_state_files(state, tmp_path)
        assert (tmp_path / "plan.md").exists()
        assert (tmp_path / "plan.md").read_text() == ""


class TestReadStateFiles:
    """Tests for read_state_files function."""

    def test_reads_existing_files(self, tmp_path):
        """Test reads state from existing files."""
        (tmp_path / "plan.md").write_text("Existing plan")
        (tmp_path / "thinking.md").write_text("Existing thinking")
        (tmp_path / "enriched_draft.md").write_text("Existing draft")
        (tmp_path / "memory.md").write_text("Existing memory")

        state = read_state_files(tmp_path)
        assert state.plan == "Existing plan"
        assert state.thinking == "Existing thinking"
        assert state.draft == "Existing draft"
        assert state.memory == "Existing memory"

    def test_returns_empty_for_missing_files(self, tmp_path):
        """Test returns empty strings when files don't exist."""
        state = read_state_files(tmp_path)
        assert state.plan == ""
        assert state.thinking == ""
        assert state.draft == ""
        assert state.memory == ""

    def test_reads_iteration_from_loop_log(self, tmp_path):
        """Test reads iteration count from loop_log.jsonl."""
        log_path = tmp_path / "loop_log.jsonl"
        log_path.write_text(
            json.dumps({"iteration": 1}) + "\n" + json.dumps({"iteration": 2}) + "\n"
        )
        state = read_state_files(tmp_path)
        assert state.iteration == 2

    def test_iteration_zero_when_no_log(self, tmp_path):
        """Test iteration is 0 when no loop_log.jsonl exists."""
        state = read_state_files(tmp_path)
        assert state.iteration == 0

    def test_reads_is_complete_from_last_entry(self, tmp_path):
        """Test reads is_complete from last loop_log entry."""
        log_path = tmp_path / "loop_log.jsonl"
        log_path.write_text(
            json.dumps({"iteration": 1, "is_complete": False})
            + "\n"
            + json.dumps({"iteration": 2, "is_complete": True})
            + "\n"
        )
        state = read_state_files(tmp_path)
        assert state.is_complete is True

    def test_partial_files(self, tmp_path):
        """Test reads whatever files exist, returns empty for missing."""
        (tmp_path / "plan.md").write_text("Only plan exists")
        state = read_state_files(tmp_path)
        assert state.plan == "Only plan exists"
        assert state.thinking == ""
        assert state.draft == ""
        assert state.memory == ""


class TestParseStructuredResponse:
    """Tests for parse_structured_response function."""

    def test_parses_complete_response(self):
        """Test parsing a fully structured agent response."""
        response = """## Plan
Analyze the codebase structure.

## Thinking
The project uses a modular architecture.

## Memory
Found key patterns in src/main.py.

## Enrichment
The user's request relates to refactoring the auth module.

## Self-Assessment
Quality: 8/10
Gaps: none
Complete: yes"""
        result = parse_structured_response(response)
        assert "Analyze the codebase" in result.plan
        assert "modular architecture" in result.thinking
        assert "key patterns" in result.memory
        assert "auth module" in result.enrichment
        assert result.self_assessment.quality == 8
        assert result.self_assessment.is_complete is True

    def test_parses_incomplete_response(self):
        """Test parsing response with missing sections."""
        response = """## Enrichment
Simple enrichment output.

## Self-Assessment
Quality: 5/10
Gaps: missing context
Complete: no"""
        result = parse_structured_response(response)
        assert result.plan == ""
        assert result.thinking == ""
        assert result.memory == ""
        assert "Simple enrichment" in result.enrichment
        assert result.self_assessment.quality == 5
        assert result.self_assessment.is_complete is False

    def test_parses_quality_score(self):
        """Test quality score extraction from various formats."""
        response = """## Self-Assessment
Quality: 7/10
Gaps: some
Complete: no"""
        result = parse_structured_response(response)
        assert result.self_assessment.quality == 7

    def test_parses_complete_yes(self):
        """Test 'Complete: yes' parsed as True."""
        response = """## Self-Assessment
Quality: 9/10
Gaps: none
Complete: yes"""
        result = parse_structured_response(response)
        assert result.self_assessment.is_complete is True

    def test_parses_complete_no(self):
        """Test 'Complete: no' parsed as False."""
        response = """## Self-Assessment
Quality: 3/10
Gaps: many gaps
Complete: no"""
        result = parse_structured_response(response)
        assert result.self_assessment.is_complete is False

    def test_empty_response(self):
        """Test parsing empty response returns defaults."""
        result = parse_structured_response("")
        assert result.plan == ""
        assert result.enrichment == ""
        assert result.self_assessment.quality == 0
        assert result.self_assessment.is_complete is False

    def test_no_self_assessment(self):
        """Test parsing response without self-assessment section."""
        response = """## Enrichment
Some output here."""
        result = parse_structured_response(response)
        assert result.enrichment == "Some output here."
        assert result.self_assessment.quality == 0
        assert result.self_assessment.is_complete is False

    def test_gaps_extracted(self):
        """Test gaps text is extracted from self-assessment."""
        response = """## Self-Assessment
Quality: 6/10
Gaps: Missing error handling analysis
Complete: no"""
        result = parse_structured_response(response)
        assert "Missing error handling" in result.self_assessment.gaps


class TestLoopSystemPrompt:
    """Tests for LOOP_SYSTEM_PROMPT constant."""

    def test_contains_key_instructions(self):
        """Test system prompt contains essential instructions."""
        assert "code-aware analyst" in LOOP_SYSTEM_PROMPT
        assert "iterative" in LOOP_SYSTEM_PROMPT
        assert "Do NOT" in LOOP_SYSTEM_PROMPT

    def test_is_nonempty_string(self):
        """Test system prompt is a non-empty string."""
        assert isinstance(LOOP_SYSTEM_PROMPT, str)
        assert len(LOOP_SYSTEM_PROMPT) > 50


class TestLoopResponseFormat:
    """Tests for LOOP_RESPONSE_FORMAT constant."""

    def test_contains_all_sections(self):
        """Test response format includes all required sections."""
        assert "## Plan" in LOOP_RESPONSE_FORMAT
        assert "## Thinking" in LOOP_RESPONSE_FORMAT
        assert "## Memory" in LOOP_RESPONSE_FORMAT
        assert "## Enrichment" in LOOP_RESPONSE_FORMAT
        assert "## Self-Assessment" in LOOP_RESPONSE_FORMAT

    def test_contains_assessment_fields(self):
        """Test response format includes quality, gaps, complete fields."""
        assert "Quality:" in LOOP_RESPONSE_FORMAT
        assert "Gaps:" in LOOP_RESPONSE_FORMAT

    def test_enrichment_section_specifies_markdown(self):
        """Test enrichment section requests structured markdown, not JSON."""
        assert "NOT JSON" in LOOP_RESPONSE_FORMAT
        assert "structured markdown" in LOOP_RESPONSE_FORMAT

    def test_enrichment_section_has_subsections(self):
        """Test enrichment section includes all required subsections."""
        assert "### Summary" in LOOP_RESPONSE_FORMAT
        assert "### Key Points" in LOOP_RESPONSE_FORMAT
        assert "### Context Analysis" in LOOP_RESPONSE_FORMAT
        assert "### Relevant Code References" in LOOP_RESPONSE_FORMAT
        assert "### Suggested Investigation Areas" in LOOP_RESPONSE_FORMAT
        assert "### Atomic Task Breakdown" in LOOP_RESPONSE_FORMAT
        assert "Complete:" in LOOP_RESPONSE_FORMAT


class TestBuildIterationPrompt:
    """Tests for build_iteration_prompt function."""

    def test_first_iteration_includes_transcript(self):
        """Test first iteration prompt includes the transcript."""
        prompt = build_iteration_prompt("hello world", LoopState(), 1, 3)
        assert "hello world" in prompt

    def test_first_iteration_includes_iteration_number(self):
        """Test first iteration prompt shows iteration count."""
        prompt = build_iteration_prompt("test", LoopState(), 1, 3)
        assert "Iteration 1 of 3" in prompt

    def test_first_iteration_explore_instruction(self):
        """Test first iteration includes broad exploration instruction."""
        prompt = build_iteration_prompt("test", LoopState(), 1, 3)
        assert "first iteration" in prompt.lower()
        assert "plan" in prompt.lower()

    def test_first_iteration_includes_response_format(self):
        """Test first iteration prompt includes response format."""
        prompt = build_iteration_prompt("test", LoopState(), 1, 3)
        assert "## Self-Assessment" in prompt

    def test_subsequent_iteration_includes_previous_state(self):
        """Test iteration 2+ includes previous plan, draft, memory."""
        state = LoopState(
            iteration=1,
            plan="Existing plan",
            draft="Existing draft",
            memory="Key findings",
        )
        prompt = build_iteration_prompt("test", state, 2, 3)
        assert "Existing plan" in prompt
        assert "Existing draft" in prompt
        assert "Key findings" in prompt

    def test_subsequent_iteration_refine_instruction(self):
        """Test iteration 2+ includes refinement instruction."""
        state = LoopState(iteration=1, draft="draft")
        prompt = build_iteration_prompt("test", state, 2, 3)
        assert "refine" in prompt.lower() or "Review" in prompt

    def test_no_previous_state_on_first(self):
        """Test first iteration does not include 'Previous Plan' heading."""
        prompt = build_iteration_prompt("test", LoopState(), 1, 3)
        assert "Previous Plan" not in prompt
        assert "Previous Draft" not in prompt

    def test_max_iterations_shown(self):
        """Test prompt shows max iteration count."""
        prompt = build_iteration_prompt("test", LoopState(), 1, 5)
        assert "5" in prompt


def _make_mock_process(stdout_text: str, returncode: int = 0) -> AsyncMock:
    """Create a mock asyncio subprocess with given stdout output.

    Args:
        stdout_text: Text to return as stdout.
        returncode: Process return code.

    Returns:
        AsyncMock configured as a subprocess process.
    """
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (
        stdout_text.encode("utf-8"),
        b"",
    )
    mock_process.returncode = returncode
    return mock_process


class TestEnrichmentLoopAgent:
    """Tests for EnrichmentLoopAgent class with claude CLI subprocess."""

    @pytest.fixture
    def settings(self):
        """Create test settings with API key."""
        return NeevSettings(
            sarvam_api_key="test",
            anthropic_api_key="test-key",
        )

    @pytest.fixture
    def settings_no_key(self, monkeypatch):
        """Create test settings without API key."""
        monkeypatch.delenv("NEEV_ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("NEEV_OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        return NeevSettings(
            sarvam_api_key="test",
            anthropic_api_key="",
            _env_file=None,
        )

    def test_init_stores_settings(self, settings, tmp_path):
        """Test agent stores settings on init."""
        agent = EnrichmentLoopAgent(settings, str(tmp_path))
        assert agent.settings is settings

    def test_init_stores_scratch_path(self, settings, tmp_path):
        """Test agent stores scratch_path on init."""
        agent = EnrichmentLoopAgent(settings, str(tmp_path))
        assert agent.scratch_path == str(tmp_path)

    def test_init_default_max_iterations(self, settings, tmp_path):
        """Test default max_iterations is 3."""
        agent = EnrichmentLoopAgent(settings, str(tmp_path))
        assert agent.max_iterations == 3

    def test_init_custom_max_iterations(self, settings, tmp_path):
        """Test custom max_iterations."""
        agent = EnrichmentLoopAgent(settings, str(tmp_path), max_iterations=5)
        assert agent.max_iterations == 5

    async def test_enrich_works_without_api_key(self, settings_no_key, tmp_path, mocker):
        """Test enrich works without API key (claude CLI manages its own auth)."""
        agent = EnrichmentLoopAgent(settings_no_key, str(tmp_path), max_iterations=1)

        response_text = (
            "## Enrichment\nOutput\n\n## Self-Assessment\nQuality: 9/10\nGaps: none\nComplete: yes"
        )

        mock_subprocess = mocker.patch(
            "neev_voice.llm.enrichment_loop.asyncio.create_subprocess_exec",
            return_value=_make_mock_process(response_text),
        )

        result = await agent.enrich("some text")
        assert "Output" in result

        # API key should NOT be in env when not configured
        call_kwargs = mock_subprocess.call_args.kwargs
        assert "ANTHROPIC_API_KEY" not in call_kwargs["env"]

    async def test_enrich_runs_iterations(self, settings, tmp_path, mocker):
        """Test enrich runs expected number of iterations via claude CLI."""
        agent = EnrichmentLoopAgent(settings, str(tmp_path), max_iterations=2)

        response_text = (
            "## Plan\nAnalysis plan\n\n"
            "## Thinking\nSome thinking\n\n"
            "## Memory\nKey finding\n\n"
            "## Enrichment\nDraft output\n\n"
            "## Self-Assessment\n"
            "Quality: 5/10\nGaps: some\nComplete: no"
        )

        mock_subprocess = mocker.patch(
            "neev_voice.llm.enrichment_loop.asyncio.create_subprocess_exec",
            return_value=_make_mock_process(response_text),
        )

        await agent.enrich("test input")
        assert mock_subprocess.call_count == 2

    async def test_enrich_stops_on_complete(self, settings, tmp_path, mocker):
        """Test enrich stops early when agent marks complete."""
        agent = EnrichmentLoopAgent(settings, str(tmp_path), max_iterations=5)

        response_text = (
            "## Enrichment\nFinal output\n\n"
            "## Self-Assessment\n"
            "Quality: 9/10\nGaps: none\nComplete: yes"
        )

        mock_subprocess = mocker.patch(
            "neev_voice.llm.enrichment_loop.asyncio.create_subprocess_exec",
            return_value=_make_mock_process(response_text),
        )

        result = await agent.enrich("test input")
        assert mock_subprocess.call_count == 1
        assert "Final output" in result

    async def test_enrich_returns_draft(self, settings, tmp_path, mocker):
        """Test enrich returns the draft from the last iteration."""
        agent = EnrichmentLoopAgent(settings, str(tmp_path), max_iterations=1)

        response_text = (
            "## Enrichment\nThe enriched result\n\n"
            "## Self-Assessment\n"
            "Quality: 7/10\nGaps: minor\nComplete: no"
        )

        mocker.patch(
            "neev_voice.llm.enrichment_loop.asyncio.create_subprocess_exec",
            return_value=_make_mock_process(response_text),
        )

        result = await agent.enrich("test input")
        assert "The enriched result" in result

    async def test_enrich_writes_state_files(self, settings, tmp_path, mocker):
        """Test enrich persists state files after each iteration."""
        agent = EnrichmentLoopAgent(settings, str(tmp_path), max_iterations=1)

        response_text = (
            "## Plan\nThe plan\n\n"
            "## Thinking\nThe thinking\n\n"
            "## Memory\nThe memory\n\n"
            "## Enrichment\nThe draft\n\n"
            "## Self-Assessment\n"
            "Quality: 6/10\nGaps: some\nComplete: no"
        )

        mocker.patch(
            "neev_voice.llm.enrichment_loop.asyncio.create_subprocess_exec",
            return_value=_make_mock_process(response_text),
        )

        await agent.enrich("test input")

        assert (tmp_path / "plan.md").exists()
        assert (tmp_path / "thinking.md").exists()
        assert (tmp_path / "enriched_draft.md").exists()
        assert (tmp_path / "memory.md").exists()
        assert (tmp_path / "loop_log.jsonl").exists()

    async def test_enrich_with_context(self, settings, tmp_path, mocker):
        """Test enrich includes context in prompt piped to claude CLI."""
        agent = EnrichmentLoopAgent(settings, str(tmp_path), max_iterations=1)

        response_text = (
            "## Enrichment\nOutput\n\n## Self-Assessment\nQuality: 5/10\nGaps: x\nComplete: yes"
        )

        mock_process = _make_mock_process(response_text)
        mocker.patch(
            "neev_voice.llm.enrichment_loop.asyncio.create_subprocess_exec",
            return_value=mock_process,
        )

        await agent.enrich("the question", context="some context")

        # Verify the prompt piped to stdin contains both context and question
        communicate_call = mock_process.communicate.call_args
        piped_input = communicate_call.kwargs["input"].decode("utf-8")
        assert "some context" in piped_input
        assert "the question" in piped_input

    async def test_enrich_includes_system_prompt_in_stdin(self, settings, tmp_path, mocker):
        """Test enrich pipes LOOP_SYSTEM_PROMPT as part of stdin content."""
        agent = EnrichmentLoopAgent(settings, str(tmp_path), max_iterations=1)

        response_text = (
            "## Enrichment\nOutput\n\n## Self-Assessment\nQuality: 9/10\nGaps: none\nComplete: yes"
        )

        mock_process = _make_mock_process(response_text)
        mocker.patch(
            "neev_voice.llm.enrichment_loop.asyncio.create_subprocess_exec",
            return_value=mock_process,
        )

        await agent.enrich("test")

        piped_input = mock_process.communicate.call_args.kwargs["input"].decode("utf-8")
        assert LOOP_SYSTEM_PROMPT in piped_input

    async def test_enrich_uses_claude_cli_flags(self, settings, tmp_path, mocker):
        """Test enrich invokes claude CLI with correct flags."""
        agent = EnrichmentLoopAgent(settings, str(tmp_path), max_iterations=1)

        response_text = (
            "## Enrichment\nOutput\n\n## Self-Assessment\nQuality: 9/10\nGaps: none\nComplete: yes"
        )

        mock_subprocess = mocker.patch(
            "neev_voice.llm.enrichment_loop.asyncio.create_subprocess_exec",
            return_value=_make_mock_process(response_text),
        )

        await agent.enrich("test")

        cmd_args = mock_subprocess.call_args.args
        assert cmd_args[0] == "claude"
        assert "--dangerously-skip-permissions" in cmd_args
        assert "--output-format" in cmd_args
        assert "--mcp-config" in cmd_args

    async def test_enrich_uses_continue_on_subsequent_iterations(self, settings, tmp_path, mocker):
        """Test --continue flag is added for iterations after the first."""
        agent = EnrichmentLoopAgent(settings, str(tmp_path), max_iterations=2)

        response_text = (
            "## Enrichment\nDraft\n\n## Self-Assessment\nQuality: 5/10\nGaps: some\nComplete: no"
        )

        mock_subprocess = mocker.patch(
            "neev_voice.llm.enrichment_loop.asyncio.create_subprocess_exec",
            return_value=_make_mock_process(response_text),
        )

        await agent.enrich("test")

        # First call should NOT have --continue
        first_call_args = mock_subprocess.call_args_list[0].args
        assert "--continue" not in first_call_args

        # Second call should have --continue
        second_call_args = mock_subprocess.call_args_list[1].args
        assert "--continue" in second_call_args

    async def test_enrich_sets_api_key_in_env_when_available(self, settings, tmp_path, mocker):
        """Test enrich passes ANTHROPIC_API_KEY when configured."""
        agent = EnrichmentLoopAgent(settings, str(tmp_path), max_iterations=1)

        response_text = (
            "## Enrichment\nOutput\n\n## Self-Assessment\nQuality: 9/10\nGaps: none\nComplete: yes"
        )

        mock_subprocess = mocker.patch(
            "neev_voice.llm.enrichment_loop.asyncio.create_subprocess_exec",
            return_value=_make_mock_process(response_text),
        )

        await agent.enrich("test")

        call_kwargs = mock_subprocess.call_args.kwargs
        assert call_kwargs["env"]["ANTHROPIC_API_KEY"] == "test-key"

    async def test_enrich_handles_cli_error(self, settings, tmp_path, mocker):
        """Test enrich continues gracefully when claude CLI returns non-zero."""
        agent = EnrichmentLoopAgent(settings, str(tmp_path), max_iterations=1)

        # Return non-zero but still provide some output
        response_text = "## Enrichment\nPartial output\n\n"
        mocker.patch(
            "neev_voice.llm.enrichment_loop.asyncio.create_subprocess_exec",
            return_value=_make_mock_process(response_text, returncode=1),
        )

        result = await agent.enrich("test")
        assert "Partial output" in result
