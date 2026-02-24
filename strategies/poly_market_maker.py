"""Polymarket Market Maker — Avellaneda-Stoikov model.

Quotes both sides of prediction markets with:
- Inventory-adjusted spreads (A-S model)
- Informed flow detection (large orders → widen spread)
- Market selection (niches with $500-5000/day volume)
- Dynamic fee awareness per market
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

from connectors.polymarket_client import OrderBook, Market, PolymarketClient
from core.portfolio import Platform, Portfolio, Side
from core.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy, Signal


@dataclass
class MarketState:
    """Tracks per-market state for the market maker."""
    token_id: str
    market_id: str
    outcome: str
    inventory: float = 0.0  # net shares held (positive = long)
    mid_prices: deque = field(default_factory=lambda: deque(maxlen=100))
    last_quote_time: float = 0.0
    last_mid: float = 0.0
    fee_rate: float = 0.0

    @property
    def volatility(self) -> float:
        """Estimate volatility from recent mid price changes."""
        if len(self.mid_prices) < 5:
            return 0.05  # default 5% vol
        prices = list(self.mid_prices)
        returns = [abs(prices[i] - prices[i - 1]) / max(prices[i - 1], 0.01)
                   for i in range(1, len(prices))]
        return max(sum(returns) / len(returns), 0.005)


class PolyMarketMaker(BaseStrategy):
    """Avellaneda-Stoikov market maker for Polymarket prediction markets."""

    def __init__(self, name: str, portfolio: Portfolio, risk_manager: RiskManager,
                 config: dict, poly_client: PolymarketClient):
        super().__init__(name, portfolio, risk_manager, config)
        self.client = poly_client
        self.mm_config = config.get("market_maker", {})

        self.target_spread = self.mm_config.get("target_spread", 0.05)
        self.max_inventory = self.mm_config.get("max_inventory", 200)
        self.repost_threshold = self.mm_config.get("repost_threshold", 0.02)
        self.max_position_per_market = config.get("max_position_per_market", 100)
        self.informed_flow_threshold = 500  # USD

        # Avellaneda-Stoikov parameters
        self.gamma = 0.1  # risk aversion (higher = wider spreads)
        self.kappa = 1.5  # order arrival intensity

        self.market_states: dict[str, MarketState] = {}  # token_id -> state
        self._active_orders: dict[str, dict] = {}  # order_id -> order_data

    async def on_start(self):
        self.logger.info("Market maker starting — selecting markets...")
        markets = await self.client.fetch_active_markets()
        selected = self._select_markets(markets)
        if not selected:
            self.logger.warning("No suitable markets found for market making")
            return

        token_ids = []
        for market in selected:
            for token in market.tokens:
                tid = token["token_id"]
                token_ids.append(tid)
                self.market_states[tid] = MarketState(
                    token_id=tid,
                    market_id=market.id,
                    outcome=token.get("outcome", ""),
                    fee_rate=market.fee,
                )

        self.client.on_book_update(self._on_book_update)
        await self.client.subscribe_market(token_ids)
        self.logger.info("Market maker active on %d tokens across %d markets", len(token_ids), len(selected))

    async def on_stop(self):
        cancelled = await self.client.cancel_all_orders()
        self.logger.info("Market maker stopped, cancelled %d orders", cancelled)

    async def evaluate(self) -> list[Signal]:
        signals: list[Signal] = []
        now = time.time()

        for tid, state in self.market_states.items():
            book = self.client.get_order_book(tid)
            if not book or book.spread <= 0:
                continue

            # Update mid price history
            state.mid_prices.append(book.mid_price)
            state.last_mid = book.mid_price

            # Skip if we just quoted
            if now - state.last_quote_time < 3:
                continue

            # Skip if spread is too tight (can't profit)
            min_spread = max(self.target_spread, state.fee_rate * 2 + 0.01)
            if book.spread < min_spread:
                continue

            # Check for informed flow
            if self._detect_informed_flow(book):
                self.logger.debug("Informed flow detected on %s, skipping", tid)
                continue

            # Calculate Avellaneda-Stoikov optimal quotes
            bid_price, ask_price = self._avellaneda_stoikov(book, state)
            if bid_price is None:
                continue

            # Calculate order size
            base_size = min(self.max_position_per_market * 0.3, 50)  # conservative with $500
            size = self.risk_manager.suggest_position_size(
                self.name, base_size, volatility=state.volatility,
            )
            if size < 1:
                continue

            shares_bid = size / max(bid_price, 0.01)
            shares_ask = size / max(1 - ask_price, 0.01)

            # Bid signal (buy YES)
            if bid_price > 0.01 and state.inventory < self.max_inventory:
                signals.append(Signal(
                    strategy=self.name,
                    market_id=state.market_id,
                    symbol=f"{state.outcome}",
                    direction="buy",
                    size_usd=size,
                    price=bid_price,
                    confidence=0.6,
                    metadata={"platform": "polymarket", "token_id": tid, "shares": shares_bid},
                ))

            # Ask signal (sell YES)
            if ask_price < 0.99 and state.inventory > -self.max_inventory:
                signals.append(Signal(
                    strategy=self.name,
                    market_id=state.market_id,
                    symbol=f"{state.outcome}",
                    direction="sell",
                    size_usd=size,
                    price=ask_price,
                    confidence=0.6,
                    metadata={"platform": "polymarket", "token_id": tid, "shares": shares_ask},
                ))

            state.last_quote_time = now

        return signals

    def _avellaneda_stoikov(self, book: OrderBook, state: MarketState) -> tuple[float | None, float | None]:
        """Calculate optimal bid/ask prices using Avellaneda-Stoikov model.

        reservation_price = mid - q * gamma * sigma^2 * T
        optimal_spread = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/kappa)

        Where:
            q = inventory (positive = long)
            gamma = risk aversion
            sigma = volatility
            T = time remaining (normalized)
        """
        mid = book.mid_price
        sigma = state.volatility
        q = state.inventory / max(self.max_inventory, 1)  # normalize to [-1, 1]

        # Clamp mid to valid range
        if mid <= 0.02 or mid >= 0.98:
            return None, None

        # Time factor (assume 1.0 for continuous markets)
        T = 1.0

        # Reservation price (inventory-adjusted fair value)
        reservation = mid - q * self.gamma * (sigma ** 2) * T

        # Optimal spread
        spread = self.gamma * (sigma ** 2) * T + (2 / self.gamma) * math.log(1 + self.gamma / self.kappa)
        spread = max(spread, self.target_spread)  # minimum spread

        # Adjust for fees
        spread = max(spread, state.fee_rate * 2 + 0.005)

        bid = round(reservation - spread / 2, 4)
        ask = round(reservation + spread / 2, 4)

        # Clamp to valid price range [0.01, 0.99]
        bid = max(0.01, min(bid, 0.99))
        ask = max(0.01, min(ask, 0.99))

        if bid >= ask:
            return None, None

        return bid, ask

    def _detect_informed_flow(self, book: OrderBook) -> bool:
        """Detect if large orders are entering (informed trader signal)."""
        if not book.bids or not book.asks:
            return False
        top_bid_size = book.bids[0].size * book.bids[0].price
        top_ask_size = book.asks[0].size * book.asks[0].price
        return top_bid_size > self.informed_flow_threshold or top_ask_size > self.informed_flow_threshold

    def _select_markets(self, markets: list[Market]) -> list[Market]:
        """Select niche markets suitable for market making.

        Criteria: volume $500-5000/day, active, has tokens.
        """
        selected = []
        for m in markets:
            if not m.active or not m.tokens:
                continue
            if 500 <= m.volume <= 5000:
                selected.append(m)
        selected.sort(key=lambda m: m.liquidity)  # prefer lower liquidity (less competition)
        max_markets = 5  # don't spread too thin with $500
        return selected[:max_markets]

    async def _on_book_update(self, token_id: str, book: OrderBook):
        """Handle real-time book updates from WebSocket."""
        state = self.market_states.get(token_id)
        if not state:
            return
        # Check if we need to repost (market moved beyond threshold)
        if state.last_mid > 0 and abs(book.mid_price - state.last_mid) > self.repost_threshold:
            self.logger.debug(
                "Market moved %.4f → %.4f on %s, will repost",
                state.last_mid, book.mid_price, token_id,
            )
