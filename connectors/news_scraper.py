"""Google News RSS scraper for market-related headlines.

Uses the existing aiohttp session (no new dependencies).
Deduplication via content hash — skip LLM calls when news hasn't changed.
"""
from __future__ import annotations

import hashlib
import logging
import time
import xml.etree.ElementTree as ET
from html import unescape

import aiohttp

logger = logging.getLogger(__name__)

_GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


class NewsScraper:
    def __init__(self, session: aiohttp.ClientSession):
        self._session = session
        # query → (timestamp, headlines, content_hash)
        self._cache: dict[str, tuple[float, list[dict], str]] = {}
        self._cache_ttl = 600  # 10 min

    async def fetch_news(self, query: str, max_results: int = 5) -> tuple[list[dict], str]:
        """Fetch recent news headlines for a query.

        Returns (headlines, content_hash).
        headlines: [{"title": str, "source": str, "published": str}]
        content_hash: SHA-256 of sorted titles (for dedup — skip LLM if unchanged).
        """
        now = time.time()

        # Check cache
        if query in self._cache:
            ts, headlines, h = self._cache[query]
            if now - ts < self._cache_ttl:
                return headlines, h

        try:
            params = {"q": query, "hl": "en", "gl": "US", "ceid": "US:en"}
            async with self._session.get(_GOOGLE_NEWS_RSS, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    logger.warning("Google News RSS returned %d for query: %s", resp.status, query[:50])
                    return [], ""
                text = await resp.text()
        except Exception:
            logger.exception("Failed to fetch news for: %s", query[:50])
            return [], ""

        headlines = self._parse_rss(text, max_results)
        content_hash = self._hash_headlines(headlines)

        self._cache[query] = (now, headlines, content_hash)
        return headlines, content_hash

    @staticmethod
    def _parse_rss(xml_text: str, max_results: int) -> list[dict]:
        """Parse Google News RSS XML into headline dicts."""
        headlines = []
        try:
            root = ET.fromstring(xml_text)
            for item in root.iter("item"):
                if len(headlines) >= max_results:
                    break
                title_el = item.find("title")
                source_el = item.find("source")
                pub_el = item.find("pubDate")
                headlines.append({
                    "title": unescape(title_el.text) if title_el is not None and title_el.text else "",
                    "source": source_el.text if source_el is not None and source_el.text else "",
                    "published": pub_el.text if pub_el is not None and pub_el.text else "",
                })
        except ET.ParseError:
            logger.warning("Failed to parse RSS XML")
        return headlines

    @staticmethod
    def _hash_headlines(headlines: list[dict]) -> str:
        """Deterministic hash of headline titles for dedup."""
        titles = sorted(h.get("title", "") for h in headlines)
        return hashlib.sha256("|".join(titles).encode()).hexdigest()[:16]
