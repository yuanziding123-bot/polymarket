"""Telegram bot notifier. No-op when TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID
are not configured, so the rest of the system stays decoupled.

Usage:
    notifier = Notifier()
    notifier.notify("position opened: ...")
"""
from __future__ import annotations

import httpx

from config import SETTINGS
from src.utils.logger import get_logger

log = get_logger("notifier")


class Notifier:
    def __init__(self) -> None:
        self._token = SETTINGS.telegram_bot_token
        self._chat_id = SETTINGS.telegram_chat_id
        self._enabled = bool(self._token and self._chat_id)
        self._client = httpx.Client(timeout=10.0)
        if not self._enabled:
            log.info("Telegram disabled (TELEGRAM_BOT_TOKEN/CHAT_ID not set).")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def notify(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send a message to the configured chat. Returns True on success.

        Never raises — failures are logged and swallowed so a Telegram outage
        can't take down the trading loop.
        """
        if not self._enabled:
            return False
        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        try:
            r = self._client.post(
                url,
                json={"chat_id": self._chat_id, "text": message, "parse_mode": parse_mode,
                      "disable_web_page_preview": True},
            )
            if r.status_code >= 400:
                log.warning(f"Telegram send failed {r.status_code}: {r.text[:200]}")
                return False
            return True
        except Exception as exc:
            log.warning(f"Telegram send raised: {exc}")
            return False

    # ----- shorthand event helpers --------------------------------------

    def position_opened(self, *, question: str, side: str, price: float,
                        size_usdc: float, edge: float, mode: str) -> None:
        self.notify(
            f"🟢 *{mode.upper()} BUY*\n"
            f"`{question[:80]}`\n"
            f"side: {side}  px: {price:.4f}  size: ${size_usdc:.2f}  edge: {edge:+.2%}"
        )

    def position_closed(self, *, question: str, reason: str,
                        exit_price: float, pnl_pct: float, pnl_usdc: float) -> None:
        icon = "🔴" if pnl_usdc < 0 else "✅"
        self.notify(
            f"{icon} *CLOSE — {reason}*\n"
            f"`{question[:80]}`\n"
            f"exit: {exit_price:.4f}  pnl: {pnl_pct:+.2%} (${pnl_usdc:+.2f})"
        )

    def circuit_breaker(self, reason: str) -> None:
        self.notify(f"⛔ *CIRCUIT BREAKER*\n{reason}")

    def error(self, where: str, exc: Exception) -> None:
        self.notify(f"❗ *ERROR @ {where}*\n`{type(exc).__name__}: {exc}`")
