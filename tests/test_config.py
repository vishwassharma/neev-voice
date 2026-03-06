"""Tests for persistent JSON config in neev_voice.config.

Tests load_json_config, save_json_config, update_config_value,
ensure_config_file, and the JSON settings source integration.
"""

import json

import pytest
from pydantic import ValidationError

from neev_voice.config import (
    _API_KEY_FALLBACK_ENV,
    API_KEY_FIELDS,
    CONFIG_DIR,
    CONFIG_FILE,
    DEFAULT_CONFIG,
    EnrichmentVersion,
    LLMProviderType,
    NeevSettings,
    create_default_config,
    ensure_config_file,
    load_json_config,
    save_json_config,
    update_config_value,
)
from neev_voice.exceptions import NeevConfigError


class TestConfigConstants:
    """Tests for config module constants."""

    def test_config_dir_is_under_home(self):
        """Test CONFIG_DIR points to ~/.config/neev."""
        assert CONFIG_DIR.parts[-2:] == (".config", "neev")

    def test_config_file_is_voice_json(self):
        """Test CONFIG_FILE points to voice.json inside CONFIG_DIR."""
        assert CONFIG_FILE.name == "voice.json"
        assert CONFIG_FILE.parent == CONFIG_DIR


class TestLoadJsonConfig:
    """Tests for load_json_config function."""

    def test_returns_empty_dict_when_missing(self, tmp_path):
        """Test returns empty dict when config file doesn't exist."""
        result = load_json_config(tmp_path / "missing.json")
        assert result == {}

    def test_loads_valid_json(self, tmp_path):
        """Test loads and returns valid JSON contents."""
        cfg = tmp_path / "voice.json"
        cfg.write_text('{"stt_mode": "codemix", "sample_rate": 44100}')
        result = load_json_config(cfg)
        assert result == {"stt_mode": "codemix", "sample_rate": 44100}

    def test_returns_empty_dict_on_invalid_json(self, tmp_path):
        """Test returns empty dict when JSON is malformed."""
        cfg = tmp_path / "voice.json"
        cfg.write_text("{invalid json!!!")
        result = load_json_config(cfg)
        assert result == {}

    def test_returns_empty_dict_on_empty_file(self, tmp_path):
        """Test returns empty dict when file is empty."""
        cfg = tmp_path / "voice.json"
        cfg.write_text("")
        result = load_json_config(cfg)
        assert result == {}


class TestSaveJsonConfig:
    """Tests for save_json_config function."""

    def test_creates_parent_dirs(self, tmp_path):
        """Test creates parent directories if they don't exist."""
        cfg = tmp_path / "deep" / "nested" / "voice.json"
        save_json_config({"key": "value"}, cfg)
        assert cfg.exists()
        assert json.loads(cfg.read_text()) == {"key": "value"}

    def test_overwrites_existing_file(self, tmp_path):
        """Test overwrites existing config file."""
        cfg = tmp_path / "voice.json"
        save_json_config({"old": "data"}, cfg)
        save_json_config({"new": "data"}, cfg)
        assert json.loads(cfg.read_text()) == {"new": "data"}

    def test_writes_pretty_json(self, tmp_path):
        """Test writes indented JSON with trailing newline."""
        cfg = tmp_path / "voice.json"
        save_json_config({"a": 1}, cfg)
        content = cfg.read_text()
        assert content.endswith("\n")
        assert "  " in content  # indented


class TestUpdateConfigValue:
    """Tests for update_config_value function."""

    def test_set_string_value(self, tmp_path):
        """Test setting a string config value."""
        cfg = tmp_path / "voice.json"
        save_json_config({}, cfg)
        update_config_value("claude_model", "opus", cfg)
        data = json.loads(cfg.read_text())
        assert data["claude_model"] == "opus"

    def test_set_enum_value(self, tmp_path):
        """Test setting an enum config value stores the enum value string."""
        cfg = tmp_path / "voice.json"
        save_json_config({}, cfg)
        update_config_value("stt_mode", "codemix", cfg)
        data = json.loads(cfg.read_text())
        assert data["stt_mode"] == "codemix"

    def test_set_int_value(self, tmp_path):
        """Test setting an integer config value."""
        cfg = tmp_path / "voice.json"
        save_json_config({}, cfg)
        update_config_value("sample_rate", "44100", cfg)
        data = json.loads(cfg.read_text())
        assert data["sample_rate"] == 44100

    def test_set_float_value(self, tmp_path):
        """Test setting a float config value."""
        cfg = tmp_path / "voice.json"
        save_json_config({}, cfg)
        update_config_value("silence_threshold", "0.05", cfg)
        data = json.loads(cfg.read_text())
        assert data["silence_threshold"] == 0.05

    def test_invalid_key_raises_key_error(self, tmp_path):
        """Test raises KeyError for unknown setting key."""
        cfg = tmp_path / "voice.json"
        save_json_config({}, cfg)
        with pytest.raises(KeyError, match="Unknown setting 'bogus_key'"):
            update_config_value("bogus_key", "value", cfg)

    def test_invalid_enum_value_raises_config_error(self, tmp_path):
        """Test raises NeevConfigError for invalid enum value."""
        cfg = tmp_path / "voice.json"
        save_json_config({}, cfg)
        with pytest.raises(NeevConfigError, match="Invalid value 'badmode'"):
            update_config_value("stt_mode", "badmode", cfg)

    def test_set_llm_api_base_value(self, tmp_path):
        """Test setting llm_api_base config value stores correctly."""
        cfg = tmp_path / "voice.json"
        save_json_config({}, cfg)
        update_config_value("llm_api_base", "https://example.com/v1", cfg)
        data = json.loads(cfg.read_text())
        assert data["llm_api_base"] == "https://example.com/v1"

    def test_preserves_existing_values(self, tmp_path):
        """Test updating one key preserves other existing values."""
        cfg = tmp_path / "voice.json"
        save_json_config({"claude_model": "opus", "sample_rate": 44100}, cfg)
        update_config_value("claude_model", "haiku", cfg)
        data = json.loads(cfg.read_text())
        assert data["claude_model"] == "haiku"
        assert data["sample_rate"] == 44100

    def test_creates_file_if_missing(self, tmp_path):
        """Test creates config file if it doesn't exist yet."""
        cfg = tmp_path / "new" / "voice.json"
        update_config_value("claude_model", "opus", cfg)
        assert cfg.exists()
        data = json.loads(cfg.read_text())
        assert data["claude_model"] == "opus"


class TestEnsureConfigFile:
    """Tests for ensure_config_file function."""

    def test_creates_file_when_missing(self, tmp_path):
        """Test creates config file with DEFAULT_CONFIG values."""
        cfg = tmp_path / "voice.json"
        result = ensure_config_file(cfg)
        assert result == cfg
        assert cfg.exists()
        assert json.loads(cfg.read_text()) == DEFAULT_CONFIG

    def test_noop_when_file_exists(self, tmp_path):
        """Test does not overwrite existing config file."""
        cfg = tmp_path / "voice.json"
        cfg.write_text('{"existing": true}')
        ensure_config_file(cfg)
        assert json.loads(cfg.read_text()) == {"existing": True}

    def test_returns_path(self, tmp_path):
        """Test returns the config file path."""
        cfg = tmp_path / "voice.json"
        result = ensure_config_file(cfg)
        assert result == cfg


class TestNeevSettingsJsonSource:
    """Tests for NeevSettings JSON config source integration."""

    def test_init_kwargs_override_defaults(self):
        """Test init kwargs take highest priority over defaults."""
        settings = NeevSettings(claude_model="from-init")
        assert settings.claude_model == "from-init"

    def test_settings_customise_sources_returns_tuple(self):
        """Test settings_customise_sources returns a tuple of sources."""
        from pydantic_settings import JsonConfigSettingsSource

        sources = NeevSettings.settings_customise_sources(
            NeevSettings,
            init_settings=None,
            env_settings=None,
            dotenv_settings=None,
            file_secret_settings=None,
        )
        assert isinstance(sources, tuple)
        assert len(sources) == 5
        # JSON source should be the 4th
        assert isinstance(sources[3], JsonConfigSettingsSource)


class TestLLMProviderType:
    """Tests for LLMProviderType enum."""

    def test_enum_has_anthropic(self):
        """Test ANTHROPIC enum value."""
        assert LLMProviderType.ANTHROPIC.value == "anthropic"

    def test_enum_has_openrouter(self):
        """Test OPENROUTER enum value."""
        assert LLMProviderType.OPENROUTER.value == "openrouter"

    def test_enum_from_string(self):
        """Test enum can be constructed from string value."""
        assert LLMProviderType("anthropic") == LLMProviderType.ANTHROPIC
        assert LLMProviderType("openrouter") == LLMProviderType.OPENROUTER


class TestLLMProviderFields:
    """Tests for LLM provider-related fields on NeevSettings."""

    def test_default_llm_provider_is_anthropic(self):
        """Test default llm_provider is anthropic."""
        settings = NeevSettings()
        assert settings.llm_provider == LLMProviderType.ANTHROPIC

    def test_default_llm_api_base_is_empty(self):
        """Test default llm_api_base is empty string."""
        settings = NeevSettings()
        assert settings.llm_api_base == ""

    def test_openrouter_api_key_defaults_empty(self, monkeypatch):
        """Test openrouter_api_key defaults to empty string."""
        monkeypatch.delenv("NEEV_OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        settings = NeevSettings(_env_file=None)
        assert settings.openrouter_api_key == ""

    def test_llm_provider_set_to_openrouter(self):
        """Test llm_provider can be set to openrouter."""
        settings = NeevSettings(llm_provider=LLMProviderType.OPENROUTER)
        assert settings.llm_provider == LLMProviderType.OPENROUTER

    def test_llm_api_base_custom_value(self):
        """Test llm_api_base accepts a custom URL."""
        settings = NeevSettings(llm_api_base="https://openrouter.ai/api/v1")
        assert settings.llm_api_base == "https://openrouter.ai/api/v1"

    def test_openrouter_api_key_custom_value(self):
        """Test openrouter_api_key accepts a value."""
        settings = NeevSettings(openrouter_api_key="sk-or-test")
        assert settings.openrouter_api_key == "sk-or-test"


class TestEnrichmentConfigFields:
    """Tests for enrichment agent configuration fields."""

    def test_enrichment_version_defaults_to_v2(self):
        """Test enrichment_version defaults to v2."""
        settings = NeevSettings()
        assert settings.enrichment_version == EnrichmentVersion.V2

    def test_enrichment_version_set_to_v1(self):
        """Test enrichment_version can be set to v1."""
        settings = NeevSettings(enrichment_version=EnrichmentVersion.V1)
        assert settings.enrichment_version == EnrichmentVersion.V1

    def test_enrichment_version_enum_values(self):
        """Test EnrichmentVersion enum has expected values."""
        assert EnrichmentVersion.V1.value == "v1"
        assert EnrichmentVersion.V2.value == "v2"

    def test_enrichment_version_is_str(self):
        """Test EnrichmentVersion members are strings."""
        assert isinstance(EnrichmentVersion.V1, str)

    def test_enrichment_max_iterations_defaults_to_3(self):
        """Test enrichment_max_iterations defaults to 3."""
        settings = NeevSettings()
        assert settings.enrichment_max_iterations == 3

    def test_enrichment_max_iterations_custom(self):
        """Test enrichment_max_iterations accepts custom value."""
        settings = NeevSettings(enrichment_max_iterations=5)
        assert settings.enrichment_max_iterations == 5

    def test_enrichment_max_iterations_min_is_1(self):
        """Test enrichment_max_iterations rejects value below 1."""
        with pytest.raises(ValidationError):
            NeevSettings(enrichment_max_iterations=0)

    def test_enrichment_max_iterations_max_is_10(self):
        """Test enrichment_max_iterations rejects value above 10."""
        with pytest.raises(ValidationError):
            NeevSettings(enrichment_max_iterations=11)

    def test_default_config_includes_enrichment_version(self):
        """Test DEFAULT_CONFIG includes enrichment_version."""
        assert "enrichment_version" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["enrichment_version"] == "v2"

    def test_default_config_includes_enrichment_max_iterations(self):
        """Test DEFAULT_CONFIG includes enrichment_max_iterations."""
        assert "enrichment_max_iterations" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["enrichment_max_iterations"] == 3


class TestResolvedLLMProperties:
    """Tests for resolved_llm_api_key and resolved_llm_api_base properties."""

    def test_resolved_key_returns_anthropic_when_provider_anthropic(self):
        """Test resolved_llm_api_key returns anthropic key for anthropic provider."""
        settings = NeevSettings(
            anthropic_api_key="sk-ant-test",
            openrouter_api_key="sk-or-test",
            llm_provider=LLMProviderType.ANTHROPIC,
        )
        assert settings.resolved_llm_api_key == "sk-ant-test"

    def test_resolved_key_returns_openrouter_when_provider_openrouter(self):
        """Test resolved_llm_api_key returns openrouter key for openrouter provider."""
        settings = NeevSettings(
            anthropic_api_key="sk-ant-test",
            openrouter_api_key="sk-or-test",
            llm_provider=LLMProviderType.OPENROUTER,
        )
        assert settings.resolved_llm_api_key == "sk-or-test"

    def test_resolved_api_base_returns_value(self):
        """Test resolved_llm_api_base returns the configured value."""
        settings = NeevSettings(llm_api_base="https://example.com/v1")
        assert settings.resolved_llm_api_base == "https://example.com/v1"

    def test_resolved_api_base_returns_empty_when_not_set(self):
        """Test resolved_llm_api_base returns empty string when not set."""
        settings = NeevSettings()
        assert settings.resolved_llm_api_base == ""


class TestDefaultConfig:
    """Tests for DEFAULT_CONFIG dict."""

    def test_contains_expected_keys(self):
        """Test DEFAULT_CONFIG contains all non-secret config keys."""
        expected_keys = {
            "stt_provider",
            "stt_mode",
            "tts_provider",
            "sample_rate",
            "silence_threshold",
            "silence_duration",
            "key_release_timeout",
            "claude_model",
            "claude_timeout",
            "stt_max_audio_duration",
            "llm_provider",
            "llm_api_base",
            "enrichment_version",
            "enrichment_max_iterations",
        }
        assert set(DEFAULT_CONFIG.keys()) == expected_keys

    def test_excludes_api_keys(self):
        """Test DEFAULT_CONFIG does not contain API key fields."""
        assert "sarvam_api_key" not in DEFAULT_CONFIG
        assert "anthropic_api_key" not in DEFAULT_CONFIG
        assert "openrouter_api_key" not in DEFAULT_CONFIG

    def test_llm_provider_default_is_anthropic(self):
        """Test DEFAULT_CONFIG llm_provider is 'anthropic'."""
        assert DEFAULT_CONFIG["llm_provider"] == "anthropic"

    def test_llm_api_base_default_is_empty(self):
        """Test DEFAULT_CONFIG llm_api_base is empty string."""
        assert DEFAULT_CONFIG["llm_api_base"] == ""


class TestEnsureConfigFileDefaults:
    """Tests for ensure_config_file writing DEFAULT_CONFIG."""

    def test_new_file_has_all_default_keys(self, tmp_path):
        """Test newly created config file contains all DEFAULT_CONFIG keys."""
        cfg = tmp_path / "voice.json"
        ensure_config_file(cfg)
        data = json.loads(cfg.read_text())
        for key in DEFAULT_CONFIG:
            assert key in data, f"Missing key: {key}"

    def test_new_file_has_default_values(self, tmp_path):
        """Test newly created config file values match DEFAULT_CONFIG."""
        cfg = tmp_path / "voice.json"
        ensure_config_file(cfg)
        data = json.loads(cfg.read_text())
        assert data == DEFAULT_CONFIG


class TestUpdateConfigValueLLMProvider:
    """Tests for update_config_value with llm_provider enum."""

    def test_set_llm_provider_openrouter(self, tmp_path):
        """Test setting llm_provider to openrouter."""
        cfg = tmp_path / "voice.json"
        save_json_config({}, cfg)
        update_config_value("llm_provider", "openrouter", cfg)
        data = json.loads(cfg.read_text())
        assert data["llm_provider"] == "openrouter"

    def test_set_llm_provider_anthropic(self, tmp_path):
        """Test setting llm_provider to anthropic."""
        cfg = tmp_path / "voice.json"
        save_json_config({}, cfg)
        update_config_value("llm_provider", "anthropic", cfg)
        data = json.loads(cfg.read_text())
        assert data["llm_provider"] == "anthropic"

    def test_set_llm_provider_invalid_raises(self, tmp_path):
        """Test setting llm_provider to invalid value raises NeevConfigError."""
        cfg = tmp_path / "voice.json"
        save_json_config({}, cfg)
        with pytest.raises(NeevConfigError, match="Invalid value 'badprovider'"):
            update_config_value("llm_provider", "badprovider", cfg)


class TestAPIKeyFields:
    """Tests for API_KEY_FIELDS constant."""

    def test_contains_all_api_key_fields(self):
        """Test API_KEY_FIELDS contains all three API key field names."""
        assert "sarvam_api_key" in API_KEY_FIELDS
        assert "anthropic_api_key" in API_KEY_FIELDS
        assert "openrouter_api_key" in API_KEY_FIELDS

    def test_is_frozenset(self):
        """Test API_KEY_FIELDS is immutable."""
        assert isinstance(API_KEY_FIELDS, frozenset)

    def test_no_non_key_fields(self):
        """Test API_KEY_FIELDS only contains API key fields."""
        assert len(API_KEY_FIELDS) == 3


class TestUpdateConfigValueBlocksAPIKeys:
    """Tests for update_config_value blocking API key fields."""

    def test_blocks_sarvam_api_key(self, tmp_path):
        """Test setting sarvam_api_key via config set is rejected."""
        cfg = tmp_path / "voice.json"
        save_json_config({}, cfg)
        with pytest.raises(KeyError, match="must not be stored in the config file"):
            update_config_value("sarvam_api_key", "sk-test", cfg)

    def test_blocks_anthropic_api_key(self, tmp_path):
        """Test setting anthropic_api_key via config set is rejected."""
        cfg = tmp_path / "voice.json"
        save_json_config({}, cfg)
        with pytest.raises(KeyError, match="must not be stored in the config file"):
            update_config_value("anthropic_api_key", "sk-ant-test", cfg)

    def test_blocks_openrouter_api_key(self, tmp_path):
        """Test setting openrouter_api_key via config set is rejected."""
        cfg = tmp_path / "voice.json"
        save_json_config({}, cfg)
        with pytest.raises(KeyError, match="must not be stored in the config file"):
            update_config_value("openrouter_api_key", "sk-or-test", cfg)

    def test_error_message_suggests_env_var(self, tmp_path):
        """Test error message mentions the correct env var pattern."""
        cfg = tmp_path / "voice.json"
        save_json_config({}, cfg)
        with pytest.raises(KeyError, match="NEEV_SARVAM_API_KEY"):
            update_config_value("sarvam_api_key", "sk-test", cfg)

    def test_non_key_fields_still_work(self, tmp_path):
        """Test non-API-key fields can still be set."""
        cfg = tmp_path / "voice.json"
        save_json_config({}, cfg)
        update_config_value("claude_model", "opus", cfg)
        data = json.loads(cfg.read_text())
        assert data["claude_model"] == "opus"


class TestCreateDefaultConfig:
    """Tests for create_default_config function."""

    def test_creates_file_with_defaults(self, tmp_path):
        """Test creates config file populated with DEFAULT_CONFIG."""
        cfg = tmp_path / "voice.json"
        result = create_default_config(cfg)
        assert result == cfg
        assert cfg.exists()
        data = json.loads(cfg.read_text())
        assert data == DEFAULT_CONFIG

    def test_creates_parent_dirs(self, tmp_path):
        """Test creates parent directories if needed."""
        cfg = tmp_path / "deep" / "nested" / "voice.json"
        create_default_config(cfg)
        assert cfg.exists()

    def test_raises_when_file_exists(self, tmp_path):
        """Test raises FileExistsError when file exists and force=False."""
        cfg = tmp_path / "voice.json"
        cfg.write_text("{}")
        with pytest.raises(FileExistsError, match="already exists"):
            create_default_config(cfg)

    def test_force_overwrites_existing(self, tmp_path):
        """Test force=True overwrites an existing config file."""
        cfg = tmp_path / "voice.json"
        cfg.write_text('{"old": "data"}')
        create_default_config(cfg, force=True)
        data = json.loads(cfg.read_text())
        assert data == DEFAULT_CONFIG

    def test_error_message_mentions_force(self, tmp_path):
        """Test FileExistsError message mentions --force flag."""
        cfg = tmp_path / "voice.json"
        cfg.write_text("{}")
        with pytest.raises(FileExistsError, match="--force"):
            create_default_config(cfg)

    def test_excludes_api_keys(self, tmp_path):
        """Test created config does not contain API key fields."""
        cfg = tmp_path / "voice.json"
        create_default_config(cfg)
        data = json.loads(cfg.read_text())
        for key in API_KEY_FIELDS:
            assert key not in data


class TestAPIKeyFallbackEnvVars:
    """Tests for unprefixed API key env var fallbacks.

    Verifies that standard env vars (SARVAM_API_KEY, ANTHROPIC_API_KEY,
    OPENROUTER_API_KEY) are accepted as fallbacks when the NEEV_-prefixed
    versions are not set.
    """

    # --- env vars to always clean up ---
    _ALL_ENV_VARS = [
        "NEEV_SARVAM_API_KEY",
        "NEEV_ANTHROPIC_API_KEY",
        "NEEV_OPENROUTER_API_KEY",
        "SARVAM_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
    ]

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        """Remove all API-key env vars before each test."""
        for var in self._ALL_ENV_VARS:
            monkeypatch.delenv(var, raising=False)

    def test_fallback_mapping_covers_all_api_key_fields(self):
        """Test _API_KEY_FALLBACK_ENV has an entry for every API_KEY_FIELDS member."""
        assert set(_API_KEY_FALLBACK_ENV.keys()) == API_KEY_FIELDS

    def test_unprefixed_sarvam_api_key_loads(self, monkeypatch):
        """Test SARVAM_API_KEY is picked up when NEEV_SARVAM_API_KEY is absent."""
        monkeypatch.setenv("SARVAM_API_KEY", "sk-sarvam-fallback")
        settings = NeevSettings(_env_file=None)
        assert settings.sarvam_api_key == "sk-sarvam-fallback"

    def test_unprefixed_anthropic_api_key_loads(self, monkeypatch):
        """Test ANTHROPIC_API_KEY is picked up when NEEV_ANTHROPIC_API_KEY is absent."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fallback")
        settings = NeevSettings(_env_file=None)
        assert settings.anthropic_api_key == "sk-ant-fallback"

    def test_unprefixed_openrouter_api_key_loads(self, monkeypatch):
        """Test OPENROUTER_API_KEY is picked up when NEEV_OPENROUTER_API_KEY is absent."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-fallback")
        settings = NeevSettings(_env_file=None)
        assert settings.openrouter_api_key == "sk-or-fallback"

    def test_prefixed_overrides_unprefixed(self, monkeypatch):
        """Test NEEV_-prefixed env var takes priority over unprefixed."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fallback")
        monkeypatch.setenv("NEEV_ANTHROPIC_API_KEY", "sk-ant-prefixed")
        settings = NeevSettings(_env_file=None)
        assert settings.anthropic_api_key == "sk-ant-prefixed"

    def test_fallback_ignored_when_prefixed_is_set(self, monkeypatch):
        """Test unprefixed fallback does not override an existing NEEV_ value."""
        monkeypatch.setenv("NEEV_SARVAM_API_KEY", "sk-neev-primary")
        monkeypatch.setenv("SARVAM_API_KEY", "sk-sarvam-fallback")
        settings = NeevSettings(_env_file=None)
        assert settings.sarvam_api_key == "sk-neev-primary"

    def test_prefixed_env_takes_priority_over_unprefixed(self, monkeypatch):
        """Test NEEV_-prefixed env var takes priority over unprefixed."""
        monkeypatch.setenv("CLAUDE_MODEL", "opus")
        monkeypatch.setenv("NEEV_CLAUDE_MODEL", "haiku")
        settings = NeevSettings(_env_file=None)
        assert settings.claude_model == "haiku"
