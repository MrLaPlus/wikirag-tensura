import logging
import os
import sys
from rich.logging import RichHandler


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
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
        logger.addHandler(handler)
        logger.propagate = False

    return logger
