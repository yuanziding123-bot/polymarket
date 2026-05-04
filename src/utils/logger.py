"""Loguru-based logger with file + stderr sinks."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from config import ROOT, SETTINGS

_LOG_DIR = ROOT / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logger.remove()
logger.add(sys.stderr, level=SETTINGS.log_level, enqueue=True)
logger.add(
    _LOG_DIR / "agent_{time:YYYY-MM-DD}.log",
    level=SETTINGS.log_level,
    rotation="00:00",
    retention="30 days",
    enqueue=True,
    encoding="utf-8",
)


def get_logger(name: str):
    return logger.bind(module=name)
