"""Centralized config loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _get(key: str, default: str | None = None) -> str | None:
    val = os.getenv(key, default)
    return val if val not in ("", None) else default


def _get_float(key: str, default: float) -> float:
    raw = _get(key)
    return float(raw) if raw is not None else default


def _get_int(key: str, default: int) -> int:
    raw = _get(key)
    return int(raw) if raw is not None else default


@dataclass
class Settings:
    # Polymarket
    polymarket_api_key: str | None
    polymarket_api_secret: str | None
    polymarket_api_passphrase: str | None
    polymarket_private_key: str | None
    polymarket_host: str
    polymarket_chain_id: int

    # LLM
    anthropic_api_key: str | None
    claude_model: str

    # News
    tavily_api_key: str | None

    # Runtime
    run_mode: str  # dry_run | live
    log_level: str
    bankroll_usdc: float
    max_position_fraction: float
    sqlite_path: Path

    # Scheduler
    main_loop_interval: int
    monitor_loop_interval: int

    # Circuit breaker
    max_daily_loss_pct: float
    max_total_exposure_pct: float
    max_concurrent_positions: int
    max_consecutive_losses: int
    consecutive_loss_cooldown_seconds: int

    # Telegram
    telegram_bot_token: str | None
    telegram_chat_id: str | None

    @property
    def is_live(self) -> bool:
        return self.run_mode.lower() == "live"


def load_settings() -> Settings:
    sqlite_raw = _get("SQLITE_PATH", "data/traces.db") or "data/traces.db"
    sqlite_path = (ROOT / sqlite_raw) if not Path(sqlite_raw).is_absolute() else Path(sqlite_raw)
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    return Settings(
        polymarket_api_key=_get("POLYMARKET_API_KEY"),
        polymarket_api_secret=_get("POLYMARKET_API_SECRET"),
        polymarket_api_passphrase=_get("POLYMARKET_API_PASSPHRASE"),
        polymarket_private_key=_get("POLYMARKET_PRIVATE_KEY"),
        polymarket_host=_get("POLYMARKET_HOST", "https://clob.polymarket.com"),
        polymarket_chain_id=_get_int("POLYMARKET_CHAIN_ID", 137),
        anthropic_api_key=_get("ANTHROPIC_API_KEY"),
        claude_model=_get("CLAUDE_MODEL", "claude-opus-4-7"),
        tavily_api_key=_get("TAVILY_API_KEY"),
        run_mode=_get("RUN_MODE", "dry_run") or "dry_run",
        log_level=_get("LOG_LEVEL", "INFO") or "INFO",
        bankroll_usdc=_get_float("BANKROLL_USDC", 500.0),
        max_position_fraction=_get_float("MAX_POSITION_FRACTION", 0.05),
        sqlite_path=sqlite_path,
        main_loop_interval=_get_int("MAIN_LOOP_INTERVAL_SECONDS", 600),
        monitor_loop_interval=_get_int("MONITOR_LOOP_INTERVAL_SECONDS", 30),
        max_daily_loss_pct=_get_float("MAX_DAILY_LOSS_PCT", 0.05),
        max_total_exposure_pct=_get_float("MAX_TOTAL_EXPOSURE_PCT", 0.50),
        max_concurrent_positions=_get_int("MAX_CONCURRENT_POSITIONS", 10),
        max_consecutive_losses=_get_int("MAX_CONSECUTIVE_LOSSES", 5),
        consecutive_loss_cooldown_seconds=_get_int("CONSECUTIVE_LOSS_COOLDOWN_SECONDS", 3600),
        telegram_bot_token=_get("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=_get("TELEGRAM_CHAT_ID"),
    )


SETTINGS = load_settings()
