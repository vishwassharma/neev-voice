"""Configuration module for Neev Voice.

Uses pydantic-settings to load configuration from environment variables,
.env files, and a persistent JSON config at ~/.config/neev/voice.json.

Priority (high to low): init kwargs > NEEV_* env vars > .env file > JSON config.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

CONFIG_DIR = Path.home() / ".config" / "neev"
CONFIG_FILE = CONFIG_DIR / "voice.json"

API_KEY_FIELDS: frozenset[str] = frozenset(
    {
        "sarvam_api_key",
        "anthropic_api_key",
        "openrouter_api_key",
    }
)
"""Fields that hold API keys.

These must not be stored in the JSON config file. Set them via
environment variables (NEEV_SARVAM_API_KEY, etc.) or a ``.env`` file.
"""


class LLMProviderType(str, Enum):
    """Supported LLM providers.

    Attributes:
        ANTHROPIC: Direct Anthropic API access.
        OPENROUTER: OpenRouter multi-model gateway.
    """

    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"


class STTProviderType(str, Enum):
    """Supported Speech-to-Text providers."""

    SARVAM = "sarvam"


class SarvamSTTMode(str, Enum):
    """Sarvam AI STT transcription modes.

    Attributes:
        TRANSLATE: Translate Hindi speech to English text.
        CODEMIX: Hindi-English code-mixed transcription.
        FORMAL: Formal Hindi transcription in Devanagari.
    """

    TRANSLATE = "translate"
    CODEMIX = "codemix"
    FORMAL = "formal"


class TTSProviderType(str, Enum):
    """Supported Text-to-Speech providers."""

    SARVAM = "sarvam"
    EDGE = "edge"


class NeevSettings(BaseSettings):
    """Application settings for Neev Voice.

    Settings are loaded with the following priority (high to low):
    init kwargs > NEEV_* env vars > .env file > ~/.config/neev/voice.json.

    Attributes:
        sarvam_api_key: API key for Sarvam AI services.
        stt_provider: Speech-to-text provider to use.
        stt_mode: Sarvam STT transcription mode (translate, codemix, formal).
        tts_provider: Text-to-speech provider to use.
        sample_rate: Audio sample rate in Hz.
        silence_threshold: RMS threshold below which audio is considered silence.
        silence_duration: Duration of silence in seconds to trigger end of recording.
        key_release_timeout: Seconds without spacebar to detect key release.
        claude_model: Claude model to use for LLM queries.
        claude_timeout: Timeout in seconds for Claude CLI subprocess.
        stt_max_audio_duration: Maximum audio duration in seconds per STT API call.
        anthropic_api_key: Anthropic API key for the enrichment agent.
        llm_provider: LLM provider selection (anthropic or openrouter).
        llm_api_base: Custom API base URL, empty uses provider default.
        openrouter_api_key: OpenRouter API key for multi-model gateway.
    """

    model_config = SettingsConfigDict(
        env_prefix="NEEV_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        json_file=str(CONFIG_FILE),
        extra="ignore",
    )

    sarvam_api_key: str = Field(default="", description="Sarvam AI API key")
    stt_provider: STTProviderType = Field(
        default=STTProviderType.SARVAM, description="STT provider"
    )
    stt_mode: SarvamSTTMode = Field(
        default=SarvamSTTMode.TRANSLATE,
        description="Sarvam STT mode: translate, codemix (mixed), formal (Devanagari)",
    )
    tts_provider: TTSProviderType = Field(default=TTSProviderType.EDGE, description="TTS provider")
    sample_rate: int = Field(default=16000, description="Audio sample rate in Hz")
    silence_threshold: float = Field(default=0.03, description="RMS silence detection threshold")
    silence_duration: float = Field(
        default=1.5, description="Silence duration to stop recording (seconds)"
    )
    key_release_timeout: float = Field(
        default=0.15,
        description="Seconds without spacebar input to detect key release in push-to-talk mode",
    )
    claude_model: str = Field(default="sonnet", description="Claude model name")
    claude_timeout: int = Field(default=30, description="Claude CLI timeout in seconds")
    stt_max_audio_duration: float = Field(
        default=30.0,
        description="Maximum audio duration in seconds per STT API call",
    )
    anthropic_api_key: str = Field(default="", description="Anthropic API key for enrichment agent")
    llm_provider: LLMProviderType = Field(
        default=LLMProviderType.ANTHROPIC,
        description="LLM provider: anthropic (direct) or openrouter (multi-model gateway)",
    )
    llm_api_base: str = Field(
        default="",
        description="Custom API base URL. Empty = use provider default. Set for OpenRouter: https://openrouter.ai/api/v1",
    )
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key (used when llm_provider=openrouter)",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customise settings source priority.

        Order (high to low): init > env > dotenv > JSON > secrets.

        Args:
            settings_cls: The settings class.
            init_settings: Init kwargs source.
            env_settings: Environment variable source.
            dotenv_settings: Dotenv file source.
            file_secret_settings: Secrets directory source.

        Returns:
            Tuple of settings sources in priority order.
        """
        from pydantic_settings import JsonConfigSettingsSource

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            JsonConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    @property
    def resolved_llm_api_key(self) -> str:
        """Return the active API key based on llm_provider.

        Returns:
            The OpenRouter key when provider is openrouter, otherwise the Anthropic key.
        """
        if self.llm_provider == LLMProviderType.OPENROUTER:
            return self.openrouter_api_key
        return self.anthropic_api_key

    @property
    def resolved_llm_api_base(self) -> str:
        """Return the API base URL, or empty for provider default.

        Returns:
            The custom API base URL, or empty string for provider default.
        """
        return self.llm_api_base


def load_json_config(config_file: Path = CONFIG_FILE) -> dict[str, Any]:
    """Read the JSON config file and return its contents.

    Args:
        config_file: Path to the JSON config file.

    Returns:
        Dictionary of config values, empty dict if file is missing or invalid.
    """
    if not config_file.exists():
        return {}
    try:
        return json.loads(config_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_json_config(config: dict[str, Any], config_file: Path = CONFIG_FILE) -> None:
    """Write config dictionary to the JSON config file.

    Creates parent directories if they don't exist.

    Args:
        config: Dictionary of config values to persist.
        config_file: Path to the JSON config file.
    """
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        json.dumps(config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def update_config_value(key: str, value: str, config_file: Path = CONFIG_FILE) -> None:
    """Validate and update a single config value in the JSON config file.

    Validates that the key exists in NeevSettings model fields and coerces
    the string value to the correct type (enum, int, float, Path, str).

    Args:
        key: Setting field name (e.g. 'stt_mode').
        value: String value to set.
        config_file: Path to the JSON config file.

    Raises:
        KeyError: If key is not a valid setting field.
        ValueError: If value cannot be coerced to the expected type.
    """
    fields = NeevSettings.model_fields
    if key not in fields:
        valid_keys = sorted(fields.keys())
        raise KeyError(f"Unknown setting '{key}'. Valid settings: {', '.join(valid_keys)}")

    if key in API_KEY_FIELDS:
        raise KeyError(
            f"'{key}' is an API key and must not be stored in the config file. "
            f"Set it via environment variable (NEEV_{key.upper()}) or .env file."
        )

    field_info = fields[key]
    annotation = field_info.annotation

    # Coerce value to the correct type
    coerced: Any
    if annotation is not None and isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            coerced = annotation(value).value
        except ValueError:
            valid_values = [e.value for e in annotation]
            raise ValueError(
                f"Invalid value '{value}' for '{key}'. Valid values: {', '.join(valid_values)}"
            )
    elif annotation is int:
        coerced = int(value)
    elif annotation is float:
        coerced = float(value)
    elif annotation is Path:
        coerced = value
    else:
        coerced = value

    config = load_json_config(config_file)
    config[key] = coerced
    save_json_config(config, config_file)


DEFAULT_CONFIG: dict[str, Any] = {
    "stt_provider": "sarvam",
    "stt_mode": "translate",
    "tts_provider": "edge",
    "sample_rate": 16000,
    "silence_threshold": 0.03,
    "silence_duration": 1.5,
    "key_release_timeout": 0.15,
    "claude_model": "sonnet",
    "claude_timeout": 30,
    "stt_max_audio_duration": 30.0,
    "llm_provider": "anthropic",
    "llm_api_base": "",
}
"""Default config values written on first run.

API keys (sarvam_api_key, anthropic_api_key, openrouter_api_key) are excluded
so they remain empty and are set via env vars or ``neev config set``.
"""


def ensure_config_file(config_file: Path = CONFIG_FILE) -> Path:
    """Ensure the JSON config file exists, creating with defaults if needed.

    Creates parent directories and writes DEFAULT_CONFIG if the file
    does not exist. No-op if it already exists.

    Args:
        config_file: Path to the JSON config file.

    Returns:
        Path to the config file.
    """
    if not config_file.exists():
        save_json_config(DEFAULT_CONFIG, config_file)
    return config_file


def create_default_config(config_file: Path = CONFIG_FILE, *, force: bool = False) -> Path:
    """Create a config file populated with DEFAULT_CONFIG values.

    Writes all non-secret default settings to the JSON config file.
    API keys are excluded and must be set via environment variables.

    Args:
        config_file: Path to the JSON config file.
        force: If True, overwrite an existing file. If False and the
            file already exists, raise FileExistsError.

    Returns:
        Path to the created config file.

    Raises:
        FileExistsError: If the file exists and force is False.
    """
    if config_file.exists() and not force:
        raise FileExistsError(
            f"Config file already exists: {config_file}. Use --force to overwrite."
        )
    save_json_config(DEFAULT_CONFIG, config_file)
    return config_file
