# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-03-03

### Added

- **Custom exception hierarchy** — `NeevError` base with `NeevConfigError`, `NeevSTTError`, `NeevTTSError`, `NeevLLMError`, `RecordingCancelledError`
- **Structured logging** via `structlog` across all modules (`log.py` with JSON/console rendering)
- **`py.typed` marker** (PEP 561) for downstream type checking
- **`__all__` exports** on all source modules and subpackage `__init__.py` re-exports
- **pip-audit** security scanning in CI workflow
- **mypy** type checking in CI and pre-commit hooks
- **uv.lock** committed for reproducible builds

### Changed

- **Ruff lint rules** expanded from 5 to 11 rule sets (added UP, B, SIM, RUF, S, PERF)
- **StrEnum migration** — all `(str, Enum)` classes replaced with `StrEnum` (Python 3.12+)
- **Exception handling** — providers raise domain-specific exceptions; CLI catches both legacy and new types
- **Retry logic** — `tenacity` retry/backoff on `SarvamSTT._transcribe_single()` and `SarvamTTS.synthesize()`
- **httpx.AsyncClient** reuse in Sarvam STT and TTS providers
- **`Optional[X]`** normalized to `X | None` across all source files
- **`contextlib.suppress`** replaces `try/except/pass` patterns
- **`raise ... from e`** exception chaining (B904 bugbear rule)

### Removed

- Dead code: `llm/claude.py` (legacy subprocess approach) and its tests
- Unused `aiofiles` dependency
- Unused `_last_enriched` reference and `_context` variable in CLI

### Fixed

- `asyncio.get_event_loop()` replaced with `get_running_loop()` in recorder
- Temp file leaks in `AudioRecorder.save_wav()` and TTS providers

### Security

- All API key fields blocked from JSON config storage via `API_KEY_FIELDS` frozenset
- Bandit security rules (S) enabled in ruff lint

## [0.2.0] - 2026-03-03

### Added

- **Push-to-talk recording** with SPACEBAR hold/release, ENTER to send, ESC to cancel
- **Sarvam AI STT** provider with translate, codemix, and formal modes
- **Edge TTS** and **Sarvam Bulbul TTS** providers with factory pattern
- **EnrichmentAgent** using Claude Agent SDK with read-only codebase tools (Read, Glob, Grep)
- **Intent extraction** for Hindi-English mixed speech (problem, solution, clue, agreement, disagreement, question)
- **Document discussion** with section-by-section voice review and agreement/disagreement tracking
- **Scratch pad** for artifact persistence (audio, transcription, enriched output, discussion results)
- **`discussion-result.md`** generation consolidating insights, agreements, and disagreements
- **Provider-agnostic LLM config** with `LLMProviderType` enum (Anthropic/OpenRouter), resolved API key/base URL
- **Config system** using pydantic-settings with layered priority (init > env > .env > JSON file)
- **CLI commands**: `neev listen`, `neev discuss`, `neev config` (show/set/init/path), `neev providers`, `neev version`
- **`neev config init`** subcommand to create default config file (API keys excluded for security)
- **Audio chunking** to split recordings >30s for Sarvam API limits
- **Energy-based VAD** with RMS threshold silence detection
- **GitHub Actions CI** with test matrix across Python 3.11, 3.12, 3.13
- **GitHub Actions release** workflow triggered on `v*` tags with version verification
- **Pre-commit hooks**: ruff lint/format, trailing whitespace, secret detection, YAML/TOML validation
- 331 tests with 94% code coverage

### Security

- API keys blocked from config file storage; must use environment variables
- `detect-secrets` and `detect-private-key` pre-commit hooks
- `no-commit-to-branch` hook protecting main branch

[Unreleased]: https://github.com/vishwassharma/neev-voice/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/vishwassharma/neev-voice/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/vishwassharma/neev-voice/releases/tag/v0.2.0
