"""Tests for the logging configuration module."""

import structlog

from neev_voice.log import configure_logging, get_logger


class TestConfigureLogging:
    """Tests for configure_logging()."""

    def test_configure_logging_default(self):
        """Test configure_logging sets up structlog with console renderer by default."""
        configure_logging()

        logger = structlog.get_logger()
        assert logger is not None

    def test_configure_logging_json(self):
        """Test configure_logging with json_logs=True uses JSON renderer."""
        configure_logging(json_logs=True)

        config = structlog.get_config()
        processor_types = [type(p) for p in config["processors"]]
        assert structlog.processors.JSONRenderer in processor_types

    def test_configure_logging_console(self):
        """Test configure_logging with json_logs=False uses console renderer."""
        configure_logging(json_logs=False)

        config = structlog.get_config()
        processor_types = [type(p) for p in config["processors"]]
        assert structlog.dev.ConsoleRenderer in processor_types

    def test_configure_logging_includes_log_level(self):
        """Test that configured processors include add_log_level."""
        configure_logging()

        config = structlog.get_config()
        processors = config["processors"]
        # add_log_level is a function, not a class instance
        assert structlog.processors.add_log_level in processors

    def test_configure_logging_includes_timestamper(self):
        """Test that configured processors include TimeStamper."""
        configure_logging()

        config = structlog.get_config()
        processor_types = [type(p) for p in config["processors"]]
        assert structlog.processors.TimeStamper in processor_types

    def test_configure_logging_caches_loggers(self):
        """Test that configure_logging enables logger caching."""
        configure_logging()

        config = structlog.get_config()
        assert config["cache_logger_on_first_use"] is True


class TestGetLogger:
    """Tests for get_logger()."""

    def test_get_logger_returns_logger(self):
        """Test get_logger returns a bound logger instance."""
        configure_logging()
        logger = get_logger()
        assert logger is not None

    def test_get_logger_with_name(self):
        """Test get_logger binds the given name."""
        configure_logging()
        logger = get_logger("mymodule")
        assert logger is not None

    def test_get_logger_different_names_return_different_loggers(self):
        """Test get_logger with different names returns distinct loggers."""
        configure_logging()
        logger1 = get_logger("module_a")
        logger2 = get_logger("module_b")
        # They should be separate instances
        assert logger1 is not logger2
