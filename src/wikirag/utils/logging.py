import logging
import os
import re
import sys
from rich.logging import RichHandler


_SECRET_PATTERNS = (
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._~+/=-]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"((?:api[_-]?key|token|secret|password)\s*[=:]\s*)[^\s,;]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"\b(sk-[A-Za-z0-9_-]{8,}|sk-or-v1-[A-Za-z0-9_-]{8,})\b"), "[REDACTED]"),
)


def redact_secrets(value: object) -> str:
    """Remove common API-key/token shapes before text reaches a log handler."""
    text = str(value)
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class SecretRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Materialize the formatted message so secrets in either msg or args are covered.
        record.msg = redact_secrets(record.getMessage())
        record.args = ()
        return True


def get_logger(name: str = "wikirag") -> logging.Logger:
    """Configures and returns a rich, structured logger.
    
    Using RichHandler ensures ANSI color highlights, timestamping,
    and clean stack trace formatting without cluttering the console.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_str, logging.INFO)
        logger.setLevel(level)

        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=True,
        )
        handler.addFilter(SecretRedactionFilter())
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
        logger.propagate = False

    return logger
