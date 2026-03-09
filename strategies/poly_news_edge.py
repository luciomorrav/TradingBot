"""News Edge strategy — directional bets when Claude Haiku detects mispricing.

Architecture:
  Every 60 min: refresh market shortlist (top N by volume, binary, no sports)
  Every 10 min (evaluate cycle):
    1. Check existing positions for TP/SL/timeout exits
    2. For each market: scrape news, dedup by hash, call LLM
    3. If edge > threshold AND confidence > min: generate BUY signal
    4. Shadow mode: log signal + Telegram notification, don't execute

Safety:
  - BUY only for entry, SELL only to close existing positions (no naked shorts)
  - Strategy cap: 10% of equity (hard limit on total NE exposure)
  - TP 20%, SL 15%, timeout 48h
  - News dedup: skip LLM if headlines unchanged since last analysis
  - Shadow mode first: validate 5-7 days before live
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from connectors.llm_client import LLMClient
from connectors.news_scraper import NewsScraper
from connectors.polymarket_client import Market, PolymarketClient
from core.portfolio import Platform
from strategies.base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class PolyNewsEdge(BaseStrategy):
    def __init__(
        self,
        name: str,
        portfolio,
        risk_manager,
        config: dict,
        poly_client: PolymarketClient,
        llm_client: LLMClient,
        news_scraper: NewsScraper,
    ):
        super().__init__(name, portfolio, risk_manager, config)
        self.client = poly_client
        self.llm = llm_client
        self.news = news_scraper

        ne = config.get("news_edge", {})
        self.shadow_mode = ne.get("shadow_mode", True)
        self.edge_threshold = ne.get("edge_threshold", 0.12)
        self.min_confidence = ne.get("min_confidence", 0.65)
        self.max_position_per_market = ne.get("max_position_per_market", 10)
        self.strategy_cap_pct = ne.get("strategy_cap_pct", 10.0)
        self.cooldown_hours = ne.get("cooldown_hours", 4)
        self.take_profit = ne.get("take_profit_pct", 0.20)
        self.stop_loss = ne.get("stop_loss_pct", 0.15)
        self.max_hold_hours = ne.get("max_hold_hours", 48)
        self.market_refresh_seconds = ne.get("market_refresh_seconds", 3600)
        self.max_markets = ne.get("max_markets", 10)
        self.ne_min_volume = ne.get("min_volume", 1000)
        self.ne_min_hours = ne.get("min_hours_to_expiry", 48)
        self.min_yes_price = ne.get("min_yes_price", 0.08)
        self.max_yes_price = ne.get("max_yes_price", 0.92)
        self.max_end_date_days = ne.get("max_end_date_days", 30)

        # Sports/noise filters — check news_edge config first, fall back to market_maker
        self._excluded_keywords = ne.get("excluded_keywords",
            config.get("market_maker", {}).get("excluded_keywords", []))
        self._excluded_categories = ne.get("excluded_categories",
            config.get("market_maker", {}).get("excluded_categories", []))

        # State
        self._markets: list[Market] = []
        self._last_refresh = 0.0
        # market_id → (timestamp, news_hash) — dedup + cooldown
        self._analyzed: dict[str, tuple[float, str]] = {}

        # Shadow portfolio — virtual positions to measure strategy quality before live
        # token_id → {question, side, entry_price, size_usd, entry_time}
        self._shadow_positions: dict[str, dict] = {}
        self._shadow_closed: list[dict] = []  # completed shadow trades

        # Shadow metrics (reset each evaluate cycle)
        self._shadow_analyzed = 0
        self._shadow_skipped_dedup = 0
        self._shadow_skipped_price = 0
        self._shadow_skipped_edge = 0
        self._shadow_signals = 0

        self._live_mode = False  # set by main.py
        self._db = None  # assigned by main.py for shadow state persistence

    async def on_start(self):
        # Restore shadow portfolio from DB before refreshing markets
        if self._db is not None and self.shadow_mode:
            saved = await self._db.load_state("ne_shadow_state")
            if saved:
                self._shadow_positions = saved.get("positions", {})
                self._shadow_closed = saved.get("closed", [])
                # Restore analyzed dict: lists → tuples (timestamp, news_hash)
                # Purge entries older than 2x cooldown — they're stale and would
                # incorrectly block fresh analysis after a long downtime.
                cutoff = time.time() - self.cooldown_hours * 2 * 3600
                self._analyzed = {
                    k: (v[0], v[1])
                    for k, v in saved.get("analyzed", {}).items()
                    if v[0] > cutoff
                }
                self.logger.info(
                    "Shadow portfolio restored: %d open positions, %d closed trades, %d analyzed",
                    len(self._shadow_positions), len(self._shadow_closed), len(self._analyzed),
                )

        await self._refresh_markets()
        # Subscribe to price updates for existing NE positions (may not be in shortlist)
        await self._subscribe_positions()
        mode_str = "shadow" if self.shadow_mode else "live"
        self.logger.info("News Edge started (%s mode), %d markets", mode_str, len(self._markets))

    async def on_stop(self):
        pass

    async def evaluate(self) -> list[Signal]:
        signals: list[Signal] = []

        # Reset shadow counters each cycle
        self._shadow_analyzed = 0
        self._shadow_skipped_dedup = 0
        self._shadow_skipped_price = 0
        self._shadow_skipped_edge = 0
        self._shadow_signals = 0

        # 1. Refresh market shortlist periodically
        if time.time() - self._last_refresh > self.market_refresh_seconds:
            await self._refresh_markets()

        # 2a. Check shadow positions for TP/SL/timeout (shadow mode only)
        if self.shadow_mode:
            await self._check_shadow_exits()

        # 2b. Check real positions for TP/SL/timeout exits
        signals.extend(self._check_exits())

        # 3. Check strategy cap before scanning for new entries
        ne_exposure = sum(
            pos.size for pos in self.portfolio.positions.values()
            if pos.strategy == self.name
        )
        cap = self.portfolio.equity * self.strategy_cap_pct / 100
        if ne_exposure >= cap:
            return signals  # exits only, no new entries

        # 4. Scan markets for entry signals
        for market in self._markets:
            sig = await self._analyze_market(market, cap - ne_exposure)
            if sig:
                signals.append(sig)
                self._shadow_signals += 1
                ne_exposure += sig.size_usd
                if ne_exposure >= cap:
                    break

        # 5. Log shadow metrics summary
        if self._shadow_analyzed > 0 or self._shadow_skipped_dedup > 0:
            self.logger.info(
                "Shadow metrics: analyzed=%d skipped_dedup=%d skipped_price=%d "
                "skipped_edge=%d signals=%d",
                self._shadow_analyzed, self._shadow_skipped_dedup,
                self._shadow_skipped_price, self._shadow_skipped_edge,
                self._shadow_signals,
            )

        return signals

    def _check_exits(self) -> list[Signal]:
        """Generate SELL signals for TP/SL/timeout on existing NE positions."""
        signals = []
        for key, pos in list(self.portfolio.positions.items()):
            if pos.strategy != self.name:
                continue
            if pos.avg_price <= 0:
                continue
            # Guard: skip exit check if no price update yet (e.g. after restart)
            if pos.current_price <= 0:
                continue

            pnl_pct = (pos.current_price - pos.avg_price) / pos.avg_price
            hold_hours = (time.time() - pos.entry_time) / 3600
            shares = pos.size / max(pos.avg_price, 0.01)

            close_reason = None
            if pnl_pct >= self.take_profit:
                close_reason = "TP"
            elif pnl_pct <= -self.stop_loss:
                close_reason = "SL"
            elif hold_hours >= self.max_hold_hours:
                close_reason = "timeout"

            if close_reason:
                price = max(pos.current_price, 0.01)
                # Use shares * price so close_position() recovers exact share count
                size_usd = shares * price
                self.logger.info(
                    "Exit %s %s: %s (PnL %.1f%%, hold %.1fh)",
                    pos.symbol, pos.market_id[:12], close_reason, pnl_pct * 100, hold_hours,
                )
                signals.append(Signal(
                    strategy=self.name,
                    market_id=pos.market_id,
                    symbol=pos.symbol,
                    direction="sell",
                    size_usd=size_usd,
                    price=price,
                    confidence=1.0,
                    metadata={
                        "platform": "polymarket",
                        "token_id": pos.market_id,
                        "shares": shares,
                        "fee": 0.0,
                        "close": True,
                        "reason": close_reason,
                    },
                ))
        return signals

    async def _check_shadow_exits(self):
        """Check shadow positions for TP/SL/timeout and record virtual PnL."""
        now = time.time()
        for token_id, pos in list(self._shadow_positions.items()):
            current_price = self._get_shadow_price(token_id)
            if current_price <= 0:
                continue

            pnl_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
            hold_hours = (now - pos["entry_time"]) / 3600

            close_reason = None
            if pnl_pct >= self.take_profit:
                close_reason = "TP"
            elif pnl_pct <= -self.stop_loss:
                close_reason = "SL"
            elif hold_hours >= self.max_hold_hours:
                close_reason = "timeout"

            if close_reason:
                pnl_usd = pos["size_usd"] * pnl_pct
                self._shadow_closed.append({
                    "side": pos["side"],
                    "question": pos["question"],
                    "entry_price": pos["entry_price"],
                    "exit_price": current_price,
                    "pnl_pct": pnl_pct,
                    "pnl_usd": pnl_usd,
                    "hold_hours": hold_hours,
                    "reason": close_reason,
                })
                del self._shadow_positions[token_id]
                self.logger.info(
                    "Shadow close [%s]: %s %s @ %.4f->%.4f (PnL %+.1f%% / $%+.2f, %.1fh)",
                    close_reason, pos["side"], pos["question"][:35],
                    pos["entry_price"], current_price,
                    pnl_pct * 100, pnl_usd, hold_hours,
                )
                self._log_shadow_metrics()
                await self._save_shadow_state()

    def _get_shadow_price(self, token_id: str) -> float:
        """Look up current token price: WS orderbook first, shortlist fallback."""
        book = self.client.get_order_book(token_id)
        if book and book.mid_price > 0:
            return book.mid_price
        for market in self._markets:
            for t in market.tokens:
                if t["token_id"] == token_id:
                    return float(t.get("price", 0))
        return 0.0

    def _log_shadow_metrics(self):
        """Log shadow portfolio performance metrics."""
        closed = self._shadow_closed
        if not closed:
            return
        wins = sum(1 for t in closed if t["pnl_pct"] > 0)
        win_rate = wins / len(closed) * 100
        avg_return = sum(t["pnl_pct"] for t in closed) / len(closed) * 100
        total_pnl = sum(t["pnl_usd"] for t in closed)
        self.logger.info(
            "Shadow portfolio: %d closed trades | win_rate=%.0f%% | avg_return=%+.1f%% | "
            "total_pnl=$%+.2f | open_positions=%d",
            len(closed), win_rate, avg_return, total_pnl, len(self._shadow_positions),
        )

    async def _save_shadow_state(self):
        """Persist shadow portfolio to DB so it survives restarts."""
        if self._db is None:
            return
        await self._db.save_state("ne_shadow_state", {
            "positions": self._shadow_positions,
            "closed": self._shadow_closed,
            # _analyzed: market_id → [timestamp, news_hash] (tuple serialized as list)
            "analyzed": {k: list(v) for k, v in self._analyzed.items()},
        })

    async def _analyze_market(self, market: Market, remaining_cap: float) -> Signal | None:
        """Analyze a single market: news → LLM → signal (or None)."""
        market_id = market.id

        # Skip if already have a real or shadow position in this market
        market_token_ids = [t["token_id"] for t in market.tokens]
        for pos in self.portfolio.positions.values():
            if pos.strategy == self.name and pos.market_id in market_token_ids:
                return None
        if self.shadow_mode and any(tid in self._shadow_positions for tid in market_token_ids):
            return None

        # Fetch news (cheap HTTP call — do before cooldown so breaking news
        # can bypass the 4h cooldown if headlines actually changed)
        headlines, news_hash = await self.news.fetch_news(market.question, max_results=5)
        if not headlines:
            return None

        # Dedup + cooldown: skip LLM if news hash unchanged OR cooldown not elapsed
        if market_id in self._analyzed:
            last_ts, last_hash = self._analyzed[market_id]
            if news_hash == last_hash:
                self._shadow_skipped_dedup += 1
                return None  # same news — no point re-analyzing
            if time.time() - last_ts < self.cooldown_hours * 3600:
                self._shadow_skipped_dedup += 1
                return None  # different news but cooldown still active

        # Pick the "Yes" outcome for analysis (binary market)
        yes_token = None
        for t in market.tokens:
            if t.get("outcome", "").lower() == "yes":
                yes_token = t
                break
        if not yes_token:
            return None

        current_price = yes_token.get("price", 0.5)
        headline_titles = [h["title"] for h in headlines if h.get("title")]

        # Call LLM
        self._shadow_analyzed += 1
        result = await self.llm.estimate_probability(
            market_question=market.question,
            outcome_name="Yes",
            current_price=current_price,
            headlines=headline_titles,
        )

        # Update analyzed timestamp + hash regardless of result
        self._analyzed[market_id] = (time.time(), news_hash)

        if not result:
            return None

        # Track LLM cost in portfolio (use locked method to avoid race)
        await self.portfolio.add_llm_cost(result.get("cost_usd", 0))

        prob = result["probability"]
        conf = result["confidence"]
        edge = prob - current_price  # positive = underpriced Yes

        self.logger.info(
            "LLM: %s → prob=%.2f conf=%.2f edge=%+.2f (market=%.2f)",
            market.question[:50], prob, conf, edge, current_price,
        )

        # Edge + confidence check
        if abs(edge) < self.edge_threshold or conf < self.min_confidence:
            self._shadow_skipped_edge += 1
            return None

        # Determine which token to buy
        if edge > 0:
            # Yes is underpriced → BUY Yes
            token = yes_token
            buy_price = current_price
        else:
            # Yes is overpriced → BUY No (Yes complement)
            no_token = None
            for t in market.tokens:
                if t.get("outcome", "").lower() == "no":
                    no_token = t
                    break
            if not no_token:
                return None
            token = no_token
            # Use complement of Yes price — No token's own "price" field is often
            # stale or from a thin order book and unreliable
            buy_price = 1 - current_price

        # Minimum price guard — skip if price is too low to be meaningful
        if buy_price < 0.02:
            self._shadow_skipped_price += 1
            self.logger.debug("Skip %s: buy_price %.4f too low", market.question[:40], buy_price)
            return None

        # Size: min of per-market cap and remaining strategy cap
        size_usd = min(self.max_position_per_market, remaining_cap)
        if size_usd < 1.0:
            return None

        metadata = {
            "platform": "polymarket",
            "token_id": token["token_id"],
            "shares": size_usd / max(buy_price, 0.01),
            "fee": 0.0,
            "edge": round(edge, 4),
            "llm_prob": round(prob, 4),
            "llm_conf": round(conf, 4),
            "reasoning": result.get("reasoning", ""),
        }

        if self.shadow_mode:
            metadata["shadow"] = True
            # Register virtual position for shadow portfolio tracking
            self._shadow_positions[token["token_id"]] = {
                "question": market.question,
                "side": token["outcome"],
                "entry_price": buy_price,
                "size_usd": size_usd,
                "entry_time": time.time(),
            }
            await self._save_shadow_state()

        self.logger.info(
            "%s signal: BUY %s %s $%.0f @ %.4f (edge %+.0f%%, conf %.0f%%)",
            "Shadow" if self.shadow_mode else "Live",
            token["outcome"], market.question[:40],
            size_usd, buy_price, edge * 100, conf * 100,
        )

        return Signal(
            strategy=self.name,
            market_id=token["token_id"],
            symbol=f"{token['outcome']} — {market.question[:40]}",
            direction="buy",
            size_usd=size_usd,
            price=buy_price,
            confidence=conf,
            metadata=metadata,
        )

    async def _refresh_markets(self):
        """Fetch and filter top markets by volume for news analysis."""
        self._last_refresh = time.time()
        try:
            all_markets = await self.client.fetch_active_markets()
        except Exception:
            self.logger.exception("Failed to refresh markets")
            return

        candidates = []
        for m in all_markets:
            if not m.active or not m.tokens:
                continue
            # Binary markets only (2 outcomes: Yes/No)
            if len(m.tokens) != 2:
                continue
            if m.volume < self.ne_min_volume:
                continue
            # Category filter
            if self._excluded_categories and m.category:
                if m.category.lower() in [c.lower() for c in self._excluded_categories]:
                    continue
            # Keyword filter
            if self._excluded_keywords:
                q_lower = m.question.lower()
                if any(kw.lower() in q_lower for kw in self._excluded_keywords):
                    continue
            # Expiry filter
            hours_left = self._hours_to_expiry(m.end_date)
            if 0 < hours_left < self.ne_min_hours:
                continue
            # Max expiry filter -- skip long-term markets (no news edge)
            max_hours = self.max_end_date_days * 24
            if hours_left > max_hours:
                continue
            # Price filter — skip obvious outcomes (Yes ≈ 0 or ≈ 1)
            yes_price = None
            for t in m.tokens:
                if t.get("outcome", "").lower() == "yes":
                    yes_price = t.get("price", 0.5)
                    break
            if yes_price is not None:
                if yes_price < self.min_yes_price or yes_price > self.max_yes_price:
                    continue
            candidates.append(m)

        # Sort by volume descending, take top N
        candidates.sort(key=lambda m: m.volume, reverse=True)
        self._markets = candidates[:self.max_markets]

        # Subscribe to WS for price updates (shortlist + any open shadow positions)
        token_ids = [t["token_id"] for m in self._markets for t in m.tokens]
        shadow_extras = [tid for tid in self._shadow_positions if tid not in token_ids]
        all_tokens = token_ids + shadow_extras
        if all_tokens:
            await self.client.subscribe_market(all_tokens)
        if shadow_extras:
            self.logger.debug("Re-subscribed %d shadow position tokens not in shortlist", len(shadow_extras))

        self.logger.info(
            "News Edge: refreshed %d markets (from %d candidates)",
            len(self._markets), len(candidates),
        )

    async def _subscribe_positions(self):
        """Subscribe to WS price updates for all existing NE positions (may not be in shortlist)."""
        token_ids = [
            pos.market_id for pos in self.portfolio.positions.values()
            if pos.strategy == self.name
        ]
        if token_ids:
            await self.client.subscribe_market(token_ids)
            self.logger.info("News Edge: subscribed %d existing position tokens", len(token_ids))

    @staticmethod
    def _hours_to_expiry(end_date: str) -> float:
        if not end_date:
            return 0.0
        try:
            dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            delta = dt - datetime.now(timezone.utc)
            return max(delta.total_seconds() / 3600, 0.0)
        except (ValueError, AttributeError):
            return 0.0
