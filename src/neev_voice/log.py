"""Structured logging configuration for neev-voice.

Provides a ``configure_logging`` function that sets up *structlog*
with either a human-friendly console renderer (development) or a
machine-parseable JSON renderer (production / containers).

Typical usage::

    from neev_voice.log import configure_logging, get_logger

    configure_logging(json_logs=False)
    log = get_logger(__name__)
    log.info("server started", port=8080)
"""

from __future__ import annotations

from typing import Any

import structlog

__all__ = ["configure_logging", "get_logger"]


def configure_logging(
    *, json_logs: bool = False, log_file: str | None = None, quiet: bool = False
) -> None:
    """Configure *structlog* for the application.

    Call once at startup (e.g. in the CLI entry-point) before any
    logging happens.

    Args:
        json_logs: When ``True`` use ``JSONRenderer`` for structured
            output suitable for log aggregation.  When ``False``
            (default) use ``ConsoleRenderer`` for colourful, readable
            terminal output.
        log_file: Optional path to write logs to a file instead of stdout.
        quiet: When ``True``, suppress all log output to stdout.
            Logs are still written to ``log_file`` if provided.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_logs:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    if quiet and not log_file:
        # Discard all output
        import io

        factory = structlog.PrintLoggerFactory(file=io.StringIO())
    elif log_file:
        factory = structlog.PrintLoggerFactory(
            file=open(log_file, "a", encoding="utf-8")  # noqa: SIM115
        )
    else:
        factory = structlog.PrintLoggerFactory()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=factory,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a *structlog* bound logger.

    Args:
        name: Optional logger name (typically ``__name__``).

    Returns:
        A configured bound logger instance.
    """
    if name is not None:
        return structlog.get_logger(name)
    return structlog.get_logger()
