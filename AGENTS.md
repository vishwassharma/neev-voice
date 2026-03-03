# AGENTS.md -- Neev Voice

Instructions for AI agents working on this codebase.

## Project

Neev Voice is a Python CLI voice agent for Hindi-English mixed speech. It records audio via push-to-talk, transcribes with Sarvam AI, extracts intent via Claude Agent SDK, and supports interactive document discussion.

## Setup

```bash
uv sync --group dev          # install all dependencies
pre-commit install           # install git hooks
cp .env.example .env         # configure API keys
```

## Build & Test

```bash
uv run pytest                # run full test suite (356 tests, >94% coverage)
uv run ruff check src/ tests/  # lint
uv run ruff format src/ tests/  # format
pre-commit run --all-files   # all pre-commit hooks
```

## Architecture

- **Source layout**: `src/neev_voice/` with subpackages `audio/`, `stt/`, `tts/`, `llm/`, `intent/`, `discussion/`
- **Tests**: `tests/` mirrors source structure, pytest with asyncio_mode=auto
- **Provider pattern**: ABC base classes (`stt/base.py`, `tts/base.py`) with factory functions
- **Config**: pydantic-settings v2 with `NeevSettings`, layered priority (init > env > .env > JSON file at `~/.config/neev/voice.json`)
- **API keys**: NEVER stored in config files. Set via `NEEV_*` env vars only. `API_KEY_FIELDS` frozenset blocks `config set` for key fields.
- **Scratch pad**: `.scratch/neev/{listen,discussion}/` for artifact persistence, timestamped flow dirs
- **LLM**: `EnrichmentAgent` in `llm/agent.py` uses `claude-agent-sdk` with read-only tools (Read, Glob, Grep). Provider-agnostic via `resolved_llm_api_key` / `resolved_llm_api_base` properties.

## Code Style

- Google Docstring format for all functions, methods, classes
- Ruff lint rules: E, F, I, N, W (line length 100)
- Ruff format: double quotes, space indent
- Tests first, maintain >80% coverage
- One file at a time
- Conventional commits: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`

## Key Files

| File | Purpose |
|------|---------|
| `src/neev_voice/cli.py` | Typer CLI commands (listen, discuss, config, providers, version) |
| `src/neev_voice/config.py` | NeevSettings, enums, DEFAULT_CONFIG, config helpers |
| `src/neev_voice/audio/keyboard.py` | Push-to-talk KeyboardMonitor (SPACEBAR/ENTER/ESC) |
| `src/neev_voice/audio/recorder.py` | AudioRecorder with VAD and push-to-talk |
| `src/neev_voice/llm/agent.py` | EnrichmentAgent with system prompt builder |
| `src/neev_voice/scratch.py` | ScratchPad class for artifact persistence |
| `src/neev_voice/intent/extractor.py` | IntentExtractor with Hindi-English indicators |
| `src/neev_voice/discussion/manager.py` | DiscussionManager for section-by-section review |
| `scripts/generate_release_notes.py` | Release notes from CHANGELOG.md or git log |

## Repository Layout

```
src/neev_voice/       # source code
tests/                # test suite (mirrors src/)
scripts/              # release notes generation
.github/workflows/    # CI and release pipelines
AGENTS.md             # agent instructions (this file)
CHANGELOG.md          # version history
pyproject.toml        # project metadata and tool config
.pre-commit-config.yaml  # pre-commit hooks
```

## Scratch Pad

Agents should use `.scratch/` for temporary working memory (plans, logs, partial results). Files in `.scratch/` are git-ignored. Follow the naming convention:

```
<type>-<context>_<YYYY-MM-DDThh-mm>.txt
```

Types: `plan`, `result`, `progress`, `success`, `rollback`

Each file should start with:

```
File Type: <Plan|Result|Progress Report|Success Summary|Rollback Summary>
Created At: <ISO 8601 timestamp>
Agent: <AgentName or WorkflowID>
Intended Use: <One-line purpose>
Expires: <ISO timestamp | end of run | N/A>
```
