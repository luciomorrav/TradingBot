from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

from core.portfolio import Portfolio, Trade

logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetrics:
    """Rolling window metrics for execution quality."""
    window_size: int = 50
    fills: deque = field(default_factory=lambda: deque(maxlen=50))
    slippages: deque = field(default_factory=lambda: deque(maxlen=50))
    latencies: deque = field(default_factory=lambda: deque(maxlen=50))

    def record(self, filled: bool, slippage: float, latency_ms: float):
        self.fills.append(1.0 if filled else 0.0)
        self.slippages.append(abs(slippage))
        self.latencies.append(latency_ms)

    @property
    def fill_rate(self) -> float:
        return sum(self.fills) / len(self.fills) if self.fills else 1.0

    @property
    def avg_slippage(self) -> float:
        return sum(self.slippages) / len(self.slippages) if self.slippages else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0


@dataclass
class RiskConfig:
    max_daily_drawdown_pct: float = 5.0
    max_position_pct: float = 20.0
    max_total_exposure_pct: float = 60.0
    consecutive_loss_cooldown: int = 3
    cooldown_minutes: int = 30
    min_fill_rate: float = 0.80
    max_slippage_pct: float = 1.0
    max_latency_seconds: float = 5.0

    @classmethod
    def from_dict(cls, d: dict) -> RiskConfig:
        exec_deg = d.get("execution_degradation", {})
        return cls(
            max_daily_drawdown_pct=d.get("max_daily_drawdown_pct", 5.0),
            max_position_pct=d.get("max_position_pct", 20.0),
            max_total_exposure_pct=d.get("max_total_exposure_pct", 60.0),
            consecutive_loss_cooldown=d.get("consecutive_loss_cooldown", 3),
            cooldown_minutes=d.get("cooldown_minutes", 30),
            min_fill_rate=exec_deg.get("min_fill_rate", 0.80),
            max_slippage_pct=exec_deg.get("max_slippage_pct", 1.0),
            max_latency_seconds=exec_deg.get("max_latency_seconds", 5.0),
        )


class RiskManager:
    """Enforces risk limits with kill switch, cooldown, and execution degradation detection."""

    def __init__(self, portfolio: Portfolio, config: RiskConfig, notify_callback=None):
        self.portfolio = portfolio
        self.config = config
        self.notify = notify_callback  # async callable(message: str)

        self.killed = False
        self.strategy_paused: dict[str, float] = {}  # strategy -> resume_timestamp
        self.consecutive_losses: dict[str, int] = {}  # strategy -> count
        self.execution_metrics: dict[str, ExecutionMetrics] = {}  # strategy -> metrics
        self.exempt_strategies: set[str] = set()  # strategies that bypass loss cooldown
        self.daily_start_equity: float = portfolio.equity
        self._lock = asyncio.Lock()

    def reset_daily(self):
        self.daily_start_equity = self.portfolio.equity
        logger.info("Daily risk reset. Start equity: $%.2f", self.daily_start_equity)

    # --- Pre-trade checks ---

    async def check_can_trade(self, strategy: str, size_usd: float) -> tuple[bool, str]:
        """Returns (allowed, reason). Check all risk limits before placing a trade."""
        if self.killed:
            return False, "Kill switch active"

        if self._is_strategy_paused(strategy):
            remaining = self.strategy_paused[strategy] - time.time()
            return False, f"Strategy {strategy} in cooldown ({remaining:.0f}s remaining)"

        # Drawdown check
        dd = self._current_drawdown_pct()
        if dd >= self.config.max_daily_drawdown_pct:
            await self._trigger_kill_switch(f"Daily drawdown {dd:.1f}% >= {self.config.max_daily_drawdown_pct}%")
            return False, "Kill switch triggered: max drawdown"

        # Position size check
        max_pos = self.portfolio.initial_capital * self.config.max_position_pct / 100
        if size_usd > max_pos:
            return False, f"Position ${size_usd:.0f} > max ${max_pos:.0f} ({self.config.max_position_pct}%)"

        # Total exposure check
        max_exposure = self.portfolio.initial_capital * self.config.max_total_exposure_pct / 100
        if self.portfolio.total_exposure + size_usd > max_exposure:
            return False, f"Total exposure would exceed ${max_exposure:.0f} ({self.config.max_total_exposure_pct}%)"

        # Cash check (accounts for pending live BUY orders)
        if size_usd > self.portfolio.available_cash:
            return False, f"Insufficient cash: ${self.portfolio.available_cash:.2f} available < ${size_usd:.2f} (reserved: ${self.portfolio.reserved_cash:.2f})"

        # Execution degradation check
        exec_check = self._check_execution_quality(strategy)
        if exec_check:
            return False, exec_check

        return True, "OK"

    def suggest_position_size(self, strategy: str, base_size: float, volatility: float = 0.0) -> float:
        """Dynamic position sizing. Reduces size when volatility is high or execution is degrading."""
        max_pos = self.portfolio.initial_capital * self.config.max_position_pct / 100
        size = min(base_size, max_pos, self.portfolio.available_cash)

        # Reduce based on volatility (higher vol = smaller position)
        if volatility > 0:
            vol_factor = max(0.3, 1.0 - volatility)  # at least 30% of base size
            size *= vol_factor

        # Reduce if execution is degrading
        metrics = self.execution_metrics.get(strategy)
        if metrics and len(metrics.fills) >= 10:
            if metrics.fill_rate < 0.9:
                size *= 0.5  # halve size if fill rate dropping
            if metrics.avg_slippage > self.config.max_slippage_pct * 0.5:
                size *= 0.7  # reduce if slippage rising

        # Don't exceed remaining exposure capacity
        remaining_capacity = (
            self.portfolio.initial_capital * self.config.max_total_exposure_pct / 100
            - self.portfolio.total_exposure
        )
        size = min(size, max(0, remaining_capacity))

        return round(size, 2)

    # --- Post-trade recording ---

    async def record_trade(self, trade: Trade, filled: bool, pnl: float = 0.0):
        """Record a trade for execution metrics and loss tracking."""
        async with self._lock:
            strategy = trade.strategy
            if strategy not in self.execution_metrics:
                self.execution_metrics[strategy] = ExecutionMetrics()

            self.execution_metrics[strategy].record(
                filled=filled,
                slippage=trade.slippage,
                latency_ms=trade.latency_ms,
            )

            # Track consecutive losses (skip for strategies that manage their own risk)
            if strategy not in self.exempt_strategies:
                if pnl < 0:
                    self.consecutive_losses[strategy] = self.consecutive_losses.get(strategy, 0) + 1
                    if self.consecutive_losses[strategy] >= self.config.consecutive_loss_cooldown:
                        await self._pause_strategy(strategy)
                elif pnl > 0:
                    self.consecutive_losses[strategy] = 0

    # --- Kill switch and cooldown ---

    async def _trigger_kill_switch(self, reason: str):
        self.killed = True
        msg = f"🚨 KILL SWITCH: {reason}"
        logger.critical(msg)
        if self.notify:
            await self.notify(msg)

    async def manual_kill(self):
        await self._trigger_kill_switch("Manual kill switch activated")

    def resume(self):
        self.killed = False
        self.strategy_paused.clear()
        logger.info("Trading resumed manually")

    async def _pause_strategy(self, strategy: str):
        resume_at = time.time() + self.config.cooldown_minutes * 60
        self.strategy_paused[strategy] = resume_at
        self.consecutive_losses[strategy] = 0
        msg = (
            f"⚠️ Strategy '{strategy}' paused for {self.config.cooldown_minutes}min "
            f"({self.config.consecutive_loss_cooldown} consecutive losses)"
        )
        logger.warning(msg)
        if self.notify:
            await self.notify(msg)

    def _is_strategy_paused(self, strategy: str) -> bool:
        if strategy not in self.strategy_paused:
            return False
        if time.time() >= self.strategy_paused[strategy]:
            del self.strategy_paused[strategy]
            return False
        return True

    # --- Internal checks ---

    def _current_drawdown_pct(self) -> float:
        if self.daily_start_equity == 0:
            return 0.0
        return max(0, (self.daily_start_equity - self.portfolio.equity) / self.daily_start_equity * 100)

    def _check_execution_quality(self, strategy: str) -> str | None:
        metrics = self.execution_metrics.get(strategy)
        if not metrics or len(metrics.fills) < 10:
            return None

        if metrics.fill_rate < self.config.min_fill_rate:
            return f"Fill rate {metrics.fill_rate:.0%} < {self.config.min_fill_rate:.0%}"

        if metrics.avg_slippage > self.config.max_slippage_pct:
            return f"Avg slippage {metrics.avg_slippage:.2f}% > {self.config.max_slippage_pct}%"

        if metrics.avg_latency_ms > self.config.max_latency_seconds * 1000:
            return f"Avg latency {metrics.avg_latency_ms:.0f}ms > {self.config.max_latency_seconds * 1000:.0f}ms"

        return None

    # --- Status ---

    def status(self) -> dict:
        return {
            "killed": self.killed,
            "daily_drawdown_pct": round(self._current_drawdown_pct(), 2),
            "paused_strategies": {
                k: round(v - time.time()) for k, v in self.strategy_paused.items() if v > time.time()
            },
            "consecutive_losses": dict(self.consecutive_losses),
            "execution_metrics": {
                k: {
                    "fill_rate": round(v.fill_rate, 2),
                    "avg_slippage": round(v.avg_slippage, 4),
                    "avg_latency_ms": round(v.avg_latency_ms, 1),
                }
                for k, v in self.execution_metrics.items()
            },
        }
