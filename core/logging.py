"""Simple logging setup (single canonical logger)."""

import logging
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format=_FORMAT)

# Export module-level logger factory
def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
