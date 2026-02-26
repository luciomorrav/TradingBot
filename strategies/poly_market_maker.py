"""Polymarket Market Maker — Avellaneda-Stoikov model.

Quotes both sides of prediction markets with:
- Inventory-adjusted spreads (A-S model)
- Informed flow detection (large orders → widen spread)
- Market selection (niches with $500-5000/day volume)
- Dynamic fee awareness per market
- True quote management: inventory from fills, order lifecycle, cancel/repost
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from connectors.polymarket_client import OrderBook, Market, PolymarketClient
from core.portfolio import Platform, Portfolio, Side, Trade
from core.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy, Signal


@dataclass
class LiveOrder:
    """Tracks a live order placed on Polymarket."""
    order_id: str
    token_id: str
    side: str        # "buy" or "sell"
    price: float
    size: float      # shares
    size_usd: float
    placed_at: float
    ttl: float = 60.0  # seconds before stale

    @property
    def is_stale(self) -> bool:
        return time.time() - self.placed_at > self.ttl


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

        self.target_spread = self.mm_config.get("target_spread", 0.02)
        self.max_inventory = self.mm_config.get("max_inventory", 200)
        self.repost_threshold = self.mm_config.get("repost_threshold", 0.02)
        self.max_position_per_market = config.get("max_position_per_market", 100)
        self.informed_flow_threshold = 500  # USD
        self.order_ttl = self.mm_config.get("order_ttl", 60.0)  # seconds
        self.min_volume = self.mm_config.get("min_volume", 200)
        self.max_volume = self.mm_config.get("max_volume", 10000)
        self.max_markets = self.mm_config.get("max_markets", 8)
        self.min_liquidity = self.mm_config.get("min_liquidity", 50)  # USD — skip empty books

        # Avellaneda-Stoikov parameters
        self.gamma = 0.1  # risk aversion (higher = wider spreads)
        self.kappa = 1.5  # order arrival intensity

        self.market_states: dict[str, MarketState] = {}  # token_id -> state
        self._active_orders: dict[str, LiveOrder] = {}    # order_id -> LiveOrder

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
        # Cancel all live orders before stopping
        await self._cancel_all_live_orders()
        self.logger.info("Market maker stopped")

    async def evaluate(self) -> list[Signal]:
        signals: list[Signal] = []
        now = time.time()

        # First: cancel stale orders and repost if market moved
        await self._manage_order_lifecycle()

        books_received = sum(1 for tid in self.market_states if self.client.get_order_book(tid))
        if books_received == 0 and self.market_states:
            self.logger.info("Waiting for orderbooks (%d tokens subscribed)", len(self.market_states))
            return signals

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

            # Skip if we already have live orders for this token
            if self._has_live_orders(tid):
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
            shares_ask = size / max(ask_price, 0.01)  # shares = usd / price

            # Inventory in USD — max_inventory is a USD cap, not a shares cap.
            # Using shares directly would block cheap tokens (e.g. 0.04) after 1 fill
            # while allowing 5-6 fills on expensive tokens (e.g. 0.96), breaking neutrality.
            inventory_usd = state.inventory * book.mid_price

            # Bid signal (buy YES) — only if inventory allows
            if bid_price > 0.01 and inventory_usd < self.max_inventory:
                signals.append(Signal(
                    strategy=self.name,
                    market_id=state.market_id,
                    symbol=f"{state.outcome}",
                    direction="buy",
                    size_usd=size,
                    price=bid_price,
                    confidence=0.6,
                    metadata={
                        "platform": "polymarket",
                        "token_id": tid,
                        "shares": shares_bid,
                        "fee": state.fee_rate * size,
                    },
                ))

            # Ask signal (sell YES) — only if inventory allows
            if ask_price < 0.99 and inventory_usd > -self.max_inventory:
                signals.append(Signal(
                    strategy=self.name,
                    market_id=state.market_id,
                    symbol=f"{state.outcome}",
                    direction="sell",
                    size_usd=size,
                    price=ask_price,
                    confidence=0.6,
                    metadata={
                        "platform": "polymarket",
                        "token_id": tid,
                        "shares": shares_ask,
                        "fee": state.fee_rate * size,
                    },
                ))

            state.last_quote_time = now

        if signals:
            self.logger.info("Generated %d signals this cycle", len(signals))
            for sig in signals[:4]:
                self.logger.info("  %s %s $%.1f @ %.4f", sig.direction, sig.symbol, sig.size_usd, sig.price)

        return signals

    # --- Order lifecycle management ---

    async def _manage_order_lifecycle(self):
        """Cancel stale orders and orders on tokens where market moved beyond threshold."""
        to_cancel = []

        for oid, order in list(self._active_orders.items()):
            # Cancel stale orders (exceeded TTL)
            if order.is_stale:
                to_cancel.append(oid)
                continue

            # Cancel if market moved beyond repost threshold
            state = self.market_states.get(order.token_id)
            if state:
                book = self.client.get_order_book(order.token_id)
                if book and state.last_mid > 0:
                    if abs(book.mid_price - state.last_mid) > self.repost_threshold:
                        to_cancel.append(oid)

        for oid in to_cancel:
            await self._cancel_order(oid)

    def _has_live_orders(self, token_id: str) -> bool:
        """Check if we already have non-stale orders for a token."""
        return any(
            o.token_id == token_id and not o.is_stale
            for o in self._active_orders.values()
        )

    async def _cancel_order(self, order_id: str):
        """Cancel a single order and remove from tracking."""
        success = await self.client.cancel_order(order_id)
        if success:
            self._active_orders.pop(order_id, None)
        else:
            self.logger.warning("Failed to cancel order %s", order_id)

    async def _cancel_all_live_orders(self):
        """Cancel all tracked orders."""
        if not self._active_orders:
            return

        await self.client.cancel_all_orders()
        count = len(self._active_orders)
        self._active_orders.clear()
        self.logger.info("Cancelled %d tracked orders", count)

    # --- Fill handling: update inventory ---

    def on_fill(self, order_id: str, filled_size: float, filled_price: float):
        """Called when an order is filled. Updates inventory from actual fills.

        This should be connected to the user WebSocket channel or
        called by the engine after execution.
        """
        order = self._active_orders.pop(order_id, None)
        if not order:
            return

        state = self.market_states.get(order.token_id)
        if not state:
            return

        # Update inventory from actual fill
        if order.side == "buy":
            state.inventory += filled_size
        else:
            state.inventory -= filled_size

        self.logger.info(
            "Fill: %s %.1f %s @ %.4f → inventory: %.1f",
            order.side, filled_size, state.outcome, filled_price, state.inventory,
        )

    def track_order(self, order_id: str, token_id: str, side: str,
                    price: float, size: float, size_usd: float):
        """Register an order for lifecycle tracking.

        Called by the engine/router after placing an order.
        """
        if order_id.startswith("paper_"):
            # In paper mode, simulate immediate fill → update inventory
            state = self.market_states.get(token_id)
            if state:
                shares = size_usd / max(price, 0.01)
                if side == "buy":
                    state.inventory += shares
                else:
                    state.inventory -= shares
            return

        self._active_orders[order_id] = LiveOrder(
            order_id=order_id,
            token_id=token_id,
            side=side,
            price=price,
            size=size,
            size_usd=size_usd,
            placed_at=time.time(),
            ttl=self.order_ttl,
        )

    # --- Avellaneda-Stoikov model ---

    def _avellaneda_stoikov(self, book: OrderBook, state: MarketState) -> tuple[float | None, float | None]:
        """Calculate optimal bid/ask prices using Avellaneda-Stoikov model.

        reservation_price = mid - q * gamma * sigma^2 * T
        optimal_spread = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/kappa)
        """
        mid = book.mid_price
        sigma = state.volatility
        q = state.inventory / max(self.max_inventory, 1)  # normalize to [-1, 1]

        # Clamp mid to valid range — exclude near-resolution markets (>95% resolved)
        if mid <= 0.05 or mid >= 0.95:
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

    @staticmethod
    def _market_is_expired(end_date: str) -> bool:
        """Return True if the market's end_date is in the past."""
        if not end_date:
            return False
        try:
            dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            return dt < datetime.now(timezone.utc)
        except (ValueError, AttributeError):
            return False

    def _select_markets(self, markets: list[Market]) -> list[Market]:
        """Select niche markets suitable for market making."""
        selected = []
        for m in markets:
            if not m.active or not m.tokens:
                continue
            if self._market_is_expired(m.end_date):
                continue
            if self.min_volume <= m.volume <= self.max_volume:
                if m.liquidity >= self.min_liquidity:
                    selected.append(m)
        selected.sort(key=lambda m: m.liquidity)  # prefer lower liquidity (less competition)
        result = selected[:self.max_markets]
        for m in result:
            self.logger.info("  Selected: %s (vol=$%.0f, liq=$%.0f)", m.question[:60], m.volume, m.liquidity)
        return result

    async def _on_book_update(self, token_id: str, book: OrderBook):
        """Handle real-time book updates from WebSocket."""
        state = self.market_states.get(token_id)
        if not state:
            return

        # Update portfolio mark-to-market so unrealized PnL is accurate
        await self.portfolio.update_price(Platform.POLYMARKET, token_id, self.name, book.mid_price)

        # Check if we need to repost (market moved beyond threshold)
        if state.last_mid > 0 and abs(book.mid_price - state.last_mid) > self.repost_threshold:
            self.logger.debug(
                "Market moved %.4f → %.4f on %s, will repost",
                state.last_mid, book.mid_price, token_id,
            )
