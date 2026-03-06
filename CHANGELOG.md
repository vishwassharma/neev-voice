# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.2] - 2026-03-06

### Changed

- **Remove dead API key passthrough in IntentClassifier** — `claude` CLI manages its own auth; removed commented-out `ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL` env passthrough code

## [0.8.1] - 2026-03-06

### Fixed

- **Streaming STT not returning responses** — removed `sample_rate` and `input_audio_codec` from WebSocket URL query params which caused Sarvam server to silently drop all responses
- **Streaming connection hangs indefinitely** — replaced `async for` receive loop with timeout-based `recv()` (15s per-message timeout) since the streaming WebSocket stays open after all responses
- **`sample_rate` type in audio message** — changed from string `"16000"` to integer `16000` to match Sarvam SDK format
- 537 tests with 95.5% code coverage

## [0.8.0] - 2026-03-05

### Added

- **WebSocket streaming STT** — audio > 30s now uses Sarvam's WebSocket streaming API (`wss://api.sarvam.ai/speech-to-text/ws`) instead of broken client-side chunking
- `_transcribe_streaming()` method on `SarvamSTT` for long audio transcription
- 10 streaming transcription tests, 4 routing tests, 5 chunk_audio edge case tests
- `websockets>=13.0` dependency

### Changed

- **`transcribe()` routing** — short audio (≤ 30s) uses REST API, long audio uses WebSocket streaming
- Removed client-side audio chunking path for long audio (was splitting words mid-sentence)
- 536 tests with 95.4% code coverage

### Fixed

- **Broken transcription for audio > 30s** — chunking at arbitrary 30-second boundaries split words/sentences, producing incoherent output

## [0.7.0] - 2026-03-04

### Added

- **`neev enrich` command** — opens `$EDITOR` for text input, then runs enrichment and intent classification (same post-transcription flow as `neev listen`, without audio/STT)
- 8 tests for the enrich command covering editor input, empty/whitespace handling, error cases, and scratch pad integration
- 520 tests with 95.3% code coverage

## [0.6.0] - 2026-03-04

### Changed

- **Remove API key requirement from `neev listen`** — the `claude` CLI manages its own auth, so `NEEV_ANTHROPIC_API_KEY` and `NEEV_OPENROUTER_API_KEY` are no longer required for listen/enrichment/classification
- **IntentClassifier** and **EnrichmentLoopAgent** now pass API keys to `claude` CLI only when configured (optional, not required)
- `neev discuss` still requires API keys (uses EnrichmentAgent v1 with `claude_agent_sdk`)
- 512 tests with 95.5% code coverage

## [0.5.0] - 2026-03-04

### Added

- **IntentClassifier** — lightweight intent classification via `claude` CLI subprocess, separate from enrichment (`intent/classifier.py`)
- 13 tests for IntentClassifier covering init, classify, discussion, error handling

### Changed

- **Separated enrichment from intent classification** in `neev listen` — enrichment agent produces structured markdown, IntentClassifier handles lightweight JSON classification
- **Enrichment output format** updated to structured markdown (not JSON) with subsections: Summary, Key Points, Context Analysis, Relevant Code References, Suggested Investigation Areas, Atomic Task Breakdown
- **`_listen_async()` flow** now calls `agent.enrich()` directly → saves to scratch → displays enrichment panel → then `classifier.classify()` for intent
- **`intent/__init__.py`** now exports `IntentClassifier`
- 512 tests with 95.4% code coverage

## [0.4.0] - 2026-03-04

### Added

- **Transcript review gate** — interactive accept/edit/reject step between transcription and enrichment (`review.py`, `TranscriptReviewer`, `TranscriptReviewAction`)
- **`--no-review` CLI flag** on `neev listen` to skip the transcript review step
- **`TranscriptRejectedError`** exception for user-rejected transcripts
- **Enrichment Agent v2 (Ralph Loop)** — iterative enrichment with self-assessment, accumulated state files, and configurable max iterations (`enrichment_loop.py`, `EnrichmentLoopAgent`)
- **`EnrichmentVersion` enum** (v1 single-pass, v2 iterative loop) and `enrichment_version` / `enrichment_max_iterations` settings
- **Enrichment agent factory** (`_get_enrichment_agent()`) in CLI for version-based agent selection
- **State file persistence** — `plan.md`, `thinking.md`, `enriched_draft.md`, `memory.md`, `loop_log.jsonl` per enrichment iteration
- **Structured response parser** for `## Plan / ## Thinking / ## Memory / ## Enrichment / ## Self-Assessment` sections
- 497 tests with 95.6% code coverage

### Changed

- **Enrichment agent v2 uses `claude` CLI** subprocess (`--dangerously-skip-permissions`, `--mcp-config`, `--continue`) instead of `claude_agent_sdk` library
- **Config table** now displays enrichment version and max iterations
- **Default enrichment version** is v2 (iterative Ralph Loop)

## [0.3.1] - 2026-03-03

### Fixed

- Disable FlakeHub in CI to avoid registration requirement for private repos
- Add `libstdc++` and `zlib` to Nix `LD_LIBRARY_PATH` so numpy/scipy C extensions load on CI
- Strip ANSI escape codes in CLI help output tests for CI compatibility

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

[Unreleased]: https://github.com/vishwassharma/neev-voice/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/vishwassharma/neev-voice/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/vishwassharma/neev-voice/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/vishwassharma/neev-voice/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/vishwassharma/neev-voice/releases/tag/v0.2.0
