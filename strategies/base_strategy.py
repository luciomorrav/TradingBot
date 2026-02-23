from __future__ import annotations

import abc
import logging
import math
from dataclasses import dataclass
from typing import Optional

from core.portfolio import Portfolio, Trade
from core.risk_manager import RiskManager


@dataclass
class Signal:
    strategy: str
    market_id: str
    symbol: str
    direction: str  # "buy" or "sell"
    size_usd: float
    price: float
    confidence: float = 1.0  # 0-1
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.direction not in ("buy", "sell"):
            raise ValueError(f"Invalid direction: {self.direction!r}, must be 'buy' or 'sell'")
        if not math.isfinite(self.price) or self.price <= 0:
            raise ValueError(f"Invalid price: {self.price}")
        if not math.isfinite(self.size_usd) or self.size_usd <= 0:
            raise ValueError(f"Invalid size_usd: {self.size_usd}")
        if not (0 <= self.confidence <= 1):
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")

    def __repr__(self):
        return (
            f"Signal(strategy={self.strategy!r}, market={self.market_id}, "
            f"{self.direction} ${self.size_usd:.2f} @ {self.price:.4f})"
        )


class BaseStrategy(abc.ABC):
    """Abstract base for all strategies. Each strategy runs independently."""

    def __init__(self, name: str, portfolio: Portfolio, risk_manager: RiskManager, config: dict):
        self.name = name
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.config = config
        self.enabled = True
        self.logger = logging.getLogger(f"strategy.{name}")

    @abc.abstractmethod
    async def on_start(self):
        """Initialize strategy (subscribe to WebSockets, load state, etc.)."""

    @abc.abstractmethod
    async def on_stop(self):
        """Clean shutdown (cancel orders, unsubscribe, etc.)."""

    @abc.abstractmethod
    async def evaluate(self) -> list[Signal]:
        """Evaluate market conditions and return trade signals (can be empty)."""

    async def run_cycle(self) -> list[Signal]:
        """Single evaluation cycle with risk checks. Called by engine."""
        if not self.enabled:
            return []
        try:
            signals = await self.evaluate()
            approved = []
            for sig in signals:
                size = self.risk_manager.suggest_position_size(self.name, sig.size_usd)
                if size < 1.0:  # minimum $1
                    continue
                sig.size_usd = size
                can_trade, reason = await self.risk_manager.check_can_trade(self.name, sig.size_usd)
                if can_trade:
                    approved.append(sig)
                else:
                    self.logger.debug("Signal rejected: %s", reason)
            return approved
        except Exception:
            self.logger.exception("Error in evaluate cycle")
            return []
