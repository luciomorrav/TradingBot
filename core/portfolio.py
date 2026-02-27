from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Platform(str, Enum):
    POLYMARKET = "polymarket"
    IB = "ib"
    KALSHI = "kalshi"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Trade:
    trade_id: str
    platform: Platform
    market_id: str
    symbol: str
    side: Side
    price: float
    size: float  # USD notional
    fee: float = 0.0
    slippage: float = 0.0  # actual vs expected price
    strategy: str = ""
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0  # time from signal to fill

    @property
    def net_cost(self) -> float:
        return self.size + self.fee


@dataclass
class Position:
    platform: Platform
    market_id: str
    symbol: str
    side: Side
    avg_price: float
    size: float  # USD notional
    strategy: str
    entry_time: float = field(default_factory=time.time)
    current_price: float = 0.0
    fees_paid: float = 0.0

    @property
    def unrealized_pnl(self) -> float:
        if self.current_price == 0:
            return 0.0
        if self.side == Side.BUY:
            return (self.current_price - self.avg_price) / self.avg_price * self.size
        return (self.avg_price - self.current_price) / self.avg_price * self.size

    @property
    def net_unrealized_pnl(self) -> float:
        return self.unrealized_pnl - self.fees_paid


class Portfolio:
    """Tracks positions, PnL, and exposure across all platforms."""

    def __init__(self, capital_usd: float):
        self.initial_capital = capital_usd
        self.cash = capital_usd
        self.positions: dict[str, Position] = {}  # key: "{platform}:{market_id}:{strategy}"
        self.closed_trades: deque[Trade] = deque(maxlen=10000)
        self.realized_pnl = 0.0
        self.total_fees = 0.0
        self.total_llm_cost = 0.0
        self._lock = asyncio.Lock()

    def _position_key(self, platform: Platform, market_id: str, strategy: str) -> str:
        return f"{platform.value}:{market_id}:{strategy}"

    async def open_position(self, trade: Trade) -> Position:
        async with self._lock:
            key = self._position_key(trade.platform, trade.market_id, trade.strategy)
            if key in self.positions:
                pos = self.positions[key]
                total_size = pos.size + trade.size
                pos.avg_price = (pos.avg_price * pos.size + trade.price * trade.size) / total_size
                pos.size = total_size
                pos.fees_paid += trade.fee
            else:
                pos = Position(
                    platform=trade.platform,
                    market_id=trade.market_id,
                    symbol=trade.symbol,
                    side=trade.side,
                    avg_price=trade.price,
                    size=trade.size,
                    strategy=trade.strategy,
                    fees_paid=trade.fee,
                )
                self.positions[key] = pos
            self.cash -= trade.net_cost
            self.total_fees += trade.fee
            return pos

    async def close_position(self, trade: Trade) -> float:
        """Close (fully or partially) a position. Returns realized PnL."""
        async with self._lock:
            key = self._position_key(trade.platform, trade.market_id, trade.strategy)
            pos = self.positions.get(key)
            if not pos:
                return 0.0

            # Work in shares to handle partial closes correctly
            trade_shares = trade.size / max(trade.price, 0.01)
            pos_shares = pos.size / max(pos.avg_price, 0.01)
            close_shares = min(trade_shares, pos_shares)

            if pos.side == Side.BUY:
                pnl = (trade.price - pos.avg_price) * close_shares
            else:
                pnl = (pos.avg_price - trade.price) * close_shares

            pnl -= trade.fee
            self.realized_pnl += pnl
            self.cash += close_shares * trade.price  # actual sale proceeds
            self.total_fees += trade.fee

            cost_basis_removed = close_shares * pos.avg_price
            pos.size -= cost_basis_removed
            if pos.size <= 0.01:  # dust threshold
                del self.positions[key]

            self.closed_trades.append(trade)
            return pnl

    async def update_price(self, platform: Platform, market_id: str, strategy: str, price: float):
        async with self._lock:
            key = self._position_key(platform, market_id, strategy)
            if key in self.positions:
                self.positions[key].current_price = price

    async def add_llm_cost(self, cost: float):
        async with self._lock:
            self.total_llm_cost += cost

    # --- Exposure and PnL queries ---

    @property
    def total_exposure(self) -> float:
        return sum(p.size for p in self.positions.values())

    @property
    def exposure_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return self.total_exposure / self.initial_capital * 100

    @property
    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.total_unrealized_pnl

    @property
    def net_pnl(self) -> float:
        """PnL net of all fees and LLM costs."""
        return self.total_pnl - self.total_llm_cost

    @property
    def equity_pnl(self) -> float:
        """Reliable PnL = equity - initial_capital. Use this for paper mode performance."""
        return self.equity - self.initial_capital

    @property
    def equity(self) -> float:
        return self.cash + self.total_exposure + self.total_unrealized_pnl

    @property
    def drawdown_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return max(0, (self.initial_capital - self.equity) / self.initial_capital * 100)

    def exposure_by_strategy(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for pos in self.positions.values():
            result[pos.strategy] = result.get(pos.strategy, 0) + pos.size
        return result

    def exposure_by_platform(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for pos in self.positions.values():
            key = pos.platform.value
            result[key] = result.get(key, 0) + pos.size
        return result

    def exposure_by_market(self) -> dict[str, float]:
        """Total exposure per market_id across all strategies."""
        result: dict[str, float] = {}
        for pos in self.positions.values():
            result[pos.market_id] = result.get(pos.market_id, 0) + pos.size
        return result

    def positions_for_strategy(self, strategy: str) -> list[Position]:
        return [p for p in self.positions.values() if p.strategy == strategy]

    def summary(self) -> dict:
        return {
            "equity": round(self.equity, 2),
            "equity_pnl": round(self.equity_pnl, 2),
            "cash": round(self.cash, 2),
            "total_exposure": round(self.total_exposure, 2),
            "exposure_pct": round(self.exposure_pct, 1),
            "realized_pnl": round(self.realized_pnl, 2),
            "unrealized_pnl": round(self.total_unrealized_pnl, 2),
            "net_pnl": round(self.net_pnl, 2),
            "drawdown_pct": round(self.drawdown_pct, 1),
            "total_fees": round(self.total_fees, 2),
            "llm_cost": round(self.total_llm_cost, 2),
            "open_positions": len(self.positions),
            "closed_trades": len(self.closed_trades),
        }

    # --- Persistence ---

    def to_dict(self) -> dict:
        """Serialize portfolio state for DB persistence."""
        return {
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "total_fees": self.total_fees,
            "total_llm_cost": self.total_llm_cost,
            "positions": {
                key: {
                    "platform": p.platform.value,
                    "market_id": p.market_id,
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "avg_price": p.avg_price,
                    "size": p.size,
                    "strategy": p.strategy,
                    "fees_paid": p.fees_paid,
                    "current_price": p.current_price,
                }
                for key, p in self.positions.items()
            },
        }

    def restore_from(self, data: dict):
        """Restore portfolio state from persisted dict. Keeps initial_capital."""
        self.cash = data.get("cash", self.initial_capital)
        self.realized_pnl = data.get("realized_pnl", 0.0)
        self.total_fees = data.get("total_fees", 0.0)
        self.total_llm_cost = data.get("total_llm_cost", 0.0)
        self.positions.clear()
        for key, p in data.get("positions", {}).items():
            self.positions[key] = Position(
                platform=Platform(p["platform"]),
                market_id=p["market_id"],
                symbol=p["symbol"],
                side=Side(p["side"]),
                avg_price=p["avg_price"],
                size=p["size"],
                strategy=p["strategy"],
                fees_paid=p.get("fees_paid", 0.0),
                current_price=p.get("current_price", 0.0),
            )
