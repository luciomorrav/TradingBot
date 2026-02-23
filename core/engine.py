from __future__ import annotations

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone

from core.portfolio import Portfolio
from core.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)


class Engine:
    """Event-driven async engine that orchestrates strategies, risk, and notifications.

    Lifecycle:
        1. Register strategies via add_strategy()
        2. Call run() — starts all strategies and the main loop
        3. Graceful shutdown via SIGINT/SIGTERM or kill switch
    """

    def __init__(self, portfolio: Portfolio, risk_manager: RiskManager, config: dict):
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.config = config
        self.mode = config.get("general", {}).get("mode", "paper")

        self.strategies: dict[str, StrategyRunner] = {}
        self.notify_callback = None  # set by telegram connector
        self.execute_callback = None  # set by connector (polymarket/ib)
        self.db_callback = None  # set by data/db.py

        self._running = False
        self._tasks: list[asyncio.Task] = []

    def add_strategy(self, strategy: BaseStrategy, interval_seconds: float = 10.0):
        runner = StrategyRunner(strategy, interval_seconds)
        self.strategies[strategy.name] = runner
        logger.info("Registered strategy: %s (interval: %.1fs)", strategy.name, interval_seconds)

    def set_notify(self, callback):
        """Set async notification callback: async def notify(message: str)"""
        self.notify_callback = callback
        self.risk_manager.notify = callback

    def set_executor(self, callback):
        """Set trade execution callback: async def execute(signal: Signal) -> Trade | None"""
        self.execute_callback = callback

    def set_db(self, callback):
        """Set DB logging callback: async def log_trade(trade: Trade, pnl: float)"""
        self.db_callback = callback

    async def run(self):
        self._running = True
        self._install_signal_handlers()

        logger.info("Engine starting in %s mode. Capital: $%.2f", self.mode, self.portfolio.initial_capital)
        if self.notify_callback:
            await self.notify_callback(
                f"🤖 Bot started ({self.mode} mode). Capital: ${self.portfolio.initial_capital:.0f}"
            )

        # Start all strategies
        for name, runner in self.strategies.items():
            await runner.strategy.on_start()
            task = asyncio.create_task(self._run_strategy_loop(runner), name=f"strategy:{name}")
            self._tasks.append(task)

        # Daily reset task
        self._tasks.append(asyncio.create_task(self._daily_reset_loop(), name="daily_reset"))

        # Heartbeat task
        self._tasks.append(asyncio.create_task(self._heartbeat_loop(), name="heartbeat"))

        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await self._shutdown()

    async def _run_strategy_loop(self, runner: StrategyRunner):
        while self._running:
            if self.risk_manager.killed:
                await asyncio.sleep(5)
                continue

            signals = await runner.strategy.run_cycle()
            for sig in signals:
                await self._process_signal(sig)

            await asyncio.sleep(runner.interval_seconds)

    async def _process_signal(self, sig: Signal):
        """Execute signal → log → notify."""
        if self.mode == "paper":
            await self._paper_execute(sig)
        elif self.execute_callback:
            trade = await self.execute_callback(sig)
            if trade:
                pnl = await self._record_trade(trade)
                await self._notify_trade(trade, pnl)
        else:
            logger.warning("No executor set, skipping signal: %s", sig)

    async def _paper_execute(self, sig: Signal):
        """Simulate execution in paper mode (instant fill at signal price)."""
        from core.portfolio import Platform, Side, Trade
        import time
        import uuid

        trade = Trade(
            trade_id=str(uuid.uuid4())[:8],
            platform=Platform.POLYMARKET,  # default, overridden by strategy metadata
            market_id=sig.market_id,
            symbol=sig.symbol,
            side=Side.BUY if sig.direction == "buy" else Side.SELL,
            price=sig.price,
            size=sig.size_usd,
            fee=0.0,  # paper mode, no fees
            slippage=0.0,
            strategy=sig.strategy,
            timestamp=time.time(),
            latency_ms=0.0,
        )
        if sig.metadata.get("platform"):
            trade.platform = Platform(sig.metadata["platform"])

        pnl = 0.0
        if sig.metadata.get("close"):
            pnl = await self.portfolio.close_position(trade)
        else:
            await self.portfolio.open_position(trade)

        await self.risk_manager.record_trade(trade, filled=True, pnl=pnl)
        await self._notify_trade(trade, pnl)

        if self.db_callback:
            await self.db_callback(trade, pnl)

    async def _record_trade(self, trade, pnl: float = 0.0):
        await self.risk_manager.record_trade(trade, filled=True, pnl=pnl)
        if self.db_callback:
            await self.db_callback(trade, pnl)
        return pnl

    async def _notify_trade(self, trade, pnl: float):
        if not self.notify_callback:
            return
        direction = "📈" if trade.side.value == "buy" else "📉"
        msg = (
            f"{direction} [{trade.strategy}] {trade.side.value.upper()} "
            f"${trade.size:.0f} {trade.symbol} @ {trade.price:.4f}"
        )
        if pnl != 0:
            msg += f" | PnL: ${pnl:+.2f}"
        await self.notify_callback(msg)

    async def _daily_reset_loop(self):
        """Reset daily risk counters at midnight UTC."""
        while self._running:
            now = datetime.now(timezone.utc)
            next_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
            if now >= next_midnight:
                from datetime import timedelta
                next_midnight += timedelta(days=1)
            seconds_until = (next_midnight - now).total_seconds()
            await asyncio.sleep(seconds_until)
            self.risk_manager.reset_daily()
            logger.info("Daily risk reset completed")

    async def _heartbeat_loop(self):
        """Send heartbeat to Telegram every 30 min."""
        while self._running:
            await asyncio.sleep(1800)
            if self.notify_callback:
                summary = self.portfolio.summary()
                await self.notify_callback(
                    f"💓 Heartbeat | Equity: ${summary['equity']:.0f} "
                    f"| PnL: ${summary['net_pnl']:+.2f} "
                    f"| Positions: {summary['open_positions']}"
                )

    async def _shutdown(self):
        logger.info("Engine shutting down...")
        self._running = False
        for _, runner in self.strategies.items():
            try:
                await runner.strategy.on_stop()
            except Exception:
                logger.exception("Error stopping strategy %s", runner.strategy.name)

        for task in self._tasks:
            if not task.done():
                task.cancel()

        if self.notify_callback:
            await self.notify_callback("🛑 Bot stopped")
        logger.info("Engine stopped")

    def _install_signal_handlers(self):
        if sys.platform != "win32":
            loop = asyncio.get_running_loop()
            for sig_name in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig_name, lambda: asyncio.create_task(self.stop()))

    async def stop(self):
        logger.info("Stop requested")
        self._running = False
        for task in self._tasks:
            if not task.done():
                task.cancel()

    # --- Status ---

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "running": self._running,
            "strategies": {
                name: {
                    "enabled": runner.strategy.enabled,
                    "interval_s": runner.interval_seconds,
                }
                for name, runner in self.strategies.items()
            },
            "portfolio": self.portfolio.summary(),
            "risk": self.risk_manager.status(),
        }


class StrategyRunner:
    def __init__(self, strategy: BaseStrategy, interval_seconds: float):
        self.strategy = strategy
        self.interval_seconds = interval_seconds
