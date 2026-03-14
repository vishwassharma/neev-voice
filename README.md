# Neev Voice

A Python CLI voice agent for Hindi-English mixed speech. Listens to user voice, transcribes it, extracts intent, discusses plan documents, and saves results to a scratch pad.

## Features

- **Push-to-talk recording** -- hold SPACEBAR to record, release to pause, ENTER to send, ESC to cancel
- **Hindi-English STT** via Sarvam AI (translate, codemix, formal modes) — REST API for short audio, WebSocket streaming for long audio (>30s)
- **Text-to-speech** with Edge TTS (free) and Sarvam Bulbul providers
- **Intent extraction** -- classifies speech into problem, solution, clue, agreement, disagreement, question
- **Interactive discuss** -- state machine for document research, TTS presentation, and voice/text enquiry with resume support
- **Enquiry-only mode** -- skip document preparation and ask questions directly (no `--files` required)
- **Enrichment agent** -- Claude Agent SDK with read-only codebase tools for structured analysis
- **Scratch pad** -- timestamped artifact directories for audio, transcriptions, enriched output, and discussion results
- **Session portability** -- export/import discuss sessions as zip archives for sharing between machines
- **Provider-agnostic LLM config** -- supports Anthropic (direct) and OpenRouter with resolved API key/base URL

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- A working microphone (for push-to-talk)
- Sarvam AI API key (for STT)
- Anthropic or OpenRouter API key (for enrichment agent)

## Installation

### As a CLI tool (recommended)

Install globally so the `neev` command is available everywhere:

```bash
# From GitHub (latest release)
uv tool install git+https://github.com/vishwassharma/neev-voice.git@v0.9.4

# From GitHub (latest main)
uv tool install git+https://github.com/vishwassharma/neev-voice.git@main

# From a local checkout
uv tool install /path/to/neev-voice
```

To update when the remote repo changes:

```bash
# If installed from a branch (e.g. @main)
uv tool upgrade neev-voice

# If pinned to a tag, reinstall with the new tag
uv tool install git+https://github.com/vishwassharma/neev-voice.git@v0.9.4 --force
```

### As a project dependency

```bash
# Add to another project
uv add git+https://github.com/vishwassharma/neev-voice.git@v0.9.4
```

### For development

```bash
# Clone the repository
git clone https://github.com/vishwassharma/neev-voice.git
cd neev-voice

# Install dependencies
uv sync --group dev

# Install pre-commit hooks
pre-commit install

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Configuration

Neev uses layered configuration (highest priority first):

1. Init kwargs (programmatic)
2. `NEEV_*` environment variables
3. `.env` file
4. `~/.config/neev/voice.json` (persistent config)

```bash
# Create a default config file
neev config init

# Show current settings
neev config

# Update a setting
neev config set stt_mode codemix
neev config set llm_provider openrouter
neev config set llm_api_base https://openrouter.ai/api/v1

# Show config file path
neev config path
```

API keys must be set via environment variables (never stored in config files):

```bash
export NEEV_SARVAM_API_KEY=your-key
export NEEV_ANTHROPIC_API_KEY=your-key
# or for OpenRouter:
export NEEV_OPENROUTER_API_KEY=your-key
```

Standard unprefixed env vars are also accepted as fallbacks, so if you already
have `ANTHROPIC_API_KEY`, `SARVAM_API_KEY`, or `OPENROUTER_API_KEY` set from
other tools, neev will pick them up automatically. The `NEEV_`-prefixed
versions always take priority when both are set.

## Usage

### Listen (record + transcribe + extract intent)

```bash
neev listen                          # default translate mode
neev listen --mode codemix           # Hindi-English mixed
neev listen --mode formal            # Devanagari output
neev listen --verbose                # show extra details
```

### Discuss (interactive research discussion)

```bash
# New session with research documents
neev discuss --files path/to/research/

# Enquiry-only mode (no documents, ask questions directly)
neev discuss

# Resume sessions
neev discuss --continue              # continue latest session
neev discuss --resume session-name   # resume by name

# Session management
neev discuss --list-sessions         # list all sessions
neev discuss --export session-name   # export session to zip
neev discuss --import path/to/zip    # import session from zip
neev discuss --migrate               # upgrade all sessions to latest schema
```

### Enrich (text enrichment without audio)

```bash
neev enrich                          # opens editor for text input
neev enrich --verbose                # with detailed output
```

### Other commands

```bash
neev version                         # show version
neev providers                       # list available STT/TTS providers
neev config                          # show configuration
neev --help                          # full help
```

## Project Structure

```
src/neev_voice/
    __init__.py          # package version
    cli.py               # Typer CLI commands
    config.py            # pydantic-settings configuration
    scratch.py           # scratch pad artifact persistence
    audio/
        keyboard.py      # push-to-talk keyboard monitor (RECORDING/PRESENTATION modes)
        recorder.py      # audio recording with VAD
    stt/
        base.py          # STT provider ABC
        sarvam.py        # Sarvam AI STT (REST + WebSocket streaming)
    tts/
        base.py          # TTS provider ABC
        edge.py          # Edge TTS + factory
        sarvam.py        # Sarvam Bulbul TTS
    llm/
        agent.py         # EnrichmentAgent (Claude Agent SDK)
    intent/
        extractor.py     # intent classification
    discussion/
        manager.py       # document discussion orchestrator (legacy)
    discuss/
        state.py         # DiscussState enum, StateSnapshot, StateStack
        session.py       # SessionManager with atomic JSON persistence
        migration.py     # versioned schema migration system
        names.py         # 3-word session name generator
        prepare.py       # PrepareEngine - Claude CLI document research
        presentation.py  # PresentationEngine - interruptible TTS playback
        enquiry.py       # EnquiryEngine - voice/text enquiry capture
        prepare_enquiry.py  # PrepareEnquiryEngine - answer research
        runner.py        # DiscussRunner - state machine orchestrator
        portability.py   # session export/import (zip archives)
tests/                   # mirrors src/ structure, 828 tests
scripts/
    generate_release_notes.py  # changelog/git-based release notes
```

## Development

```bash
# Run tests
uv run pytest

# Run linter
uv run ruff check src/ tests/

# Format code
uv run ruff format src/ tests/

# Run pre-commit on all files
pre-commit run --all-files
```

## Release Process

1. Bump version using [bump-my-version](https://github.com/callowayproject/bump-my-version):
   ```bash
   uv run bump-my-version bump patch   # 0.9.3 → 0.9.4
   uv run bump-my-version bump minor   # 0.9.3 → 0.10.0
   uv run bump-my-version bump major   # 0.9.3 → 1.0.0
   ```
   This updates `pyproject.toml` and all version references in `README.md`.
2. Add entry to `CHANGELOG.md`
3. Commit on a feature branch, merge to main
4. Tag: `git tag v0.x.x`
5. Push: `git push origin main --tags`
6. GitHub Actions builds, verifies version+changelog, and creates the release

## License

MIT
