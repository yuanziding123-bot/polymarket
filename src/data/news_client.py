"""Tavily news search wrapper. Falls back gracefully when key is missing."""
from __future__ import annotations

from dataclasses import dataclass

from config import SETTINGS
from src.utils.logger import get_logger

log = get_logger("news")


@dataclass
class NewsItem:
    title: str
    url: str
    snippet: str
    published: str | None = None


class NewsClient:
    def __init__(self) -> None:
        self._tavily = None
        if SETTINGS.tavily_api_key:
            try:
                from tavily import TavilyClient

                self._tavily = TavilyClient(api_key=SETTINGS.tavily_api_key)
            except ImportError:
                log.warning("tavily-python not installed; news disabled.")
        else:
            log.info("TAVILY_API_KEY not set; news search disabled.")

    def search(self, query: str, max_results: int = 5) -> list[NewsItem]:
        if not self._tavily:
            return []
        try:
            resp = self._tavily.search(query=query, max_results=max_results, search_depth="basic")
        except Exception as exc:
            log.warning(f"Tavily search failed for '{query[:40]}': {exc}")
            return []
        out: list[NewsItem] = []
        for r in resp.get("results", []):
            out.append(
                NewsItem(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", "")[:500],
                    published=r.get("published_date"),
                )
            )
        return out
