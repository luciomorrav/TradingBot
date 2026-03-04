from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
import uuid
from datetime import datetime, timezone

from core.portfolio import Platform, Portfolio, Side, Trade
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
        self._db = None  # Database instance for state persistence

        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._pending_orders: dict[str, dict] = {}  # order_id → {sig, token_id, side, ...}
        self._fill_lock = asyncio.Lock()  # serialize fill processing
        self._ws_connected_check = None  # callable → bool, blocks live orders if WS down
        self.poly_client = None  # set by main.py for reconciliation queries
        self._last_recon_alerts: list[str] = []  # suppress repeated identical alerts

        # Paper mode fee rate — configurable via polymarket.paper_fee_rate
        poly_cfg = config.get("polymarket", {})
        self._paper_fee_rate = poly_cfg.get("paper_fee_rate", 0.005)

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

    def set_ws_check(self, check):
        """Set callable that returns True when user WS is connected (fail-closed for live)."""
        self._ws_connected_check = check

    def set_db(self, callback, db_instance=None):
        """Set DB logging callback and optional DB instance for state persistence."""
        self.db_callback = callback
        self._db = db_instance

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

        # Pending order cleanup task (live mode)
        self._tasks.append(asyncio.create_task(self._pending_order_cleanup_loop(), name="pending_cleanup"))

        # Exchange reconciliation task (live mode)
        self._tasks.append(asyncio.create_task(self._reconciliation_loop(), name="reconciliation"))

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
        """Execute signal → log → notify → update strategy."""
        # Shadow mode: log and notify, but don't execute (for strategy validation)
        if sig.metadata.get("shadow"):
            logger.info("Shadow signal: %s %s $%.0f @ %.4f (%s)",
                        sig.direction, sig.symbol, sig.size_usd, sig.price, sig.strategy)
            if self.notify_callback:
                edge = sig.metadata.get("edge", 0)
                conf = sig.confidence
                await self.notify_callback(
                    f"\U0001f47b Shadow: {sig.direction.upper()} {sig.symbol} "
                    f"${sig.size_usd:.0f} @ {sig.price:.4f} "
                    f"(edge {edge:+.0%}, conf {conf:.0%})"
                )
            return

        if self.mode == "paper":
            await self._paper_execute(sig)
        elif self.execute_callback:
            # Fail-closed: block ALL live orders if user WS is disconnected
            if self._ws_connected_check and not self._ws_connected_check():
                logger.warning("User WS disconnected — blocking live order (%s %s $%.0f)",
                               sig.direction, sig.symbol, sig.size_usd)
                return

            # Full risk check before sending real orders (BUY adds exposure; SELL reduces it)
            if sig.direction == "buy":
                ok, reason = await self.risk_manager.check_can_trade(sig.strategy, sig.size_usd)
                if not ok:
                    logger.debug("Live signal blocked by risk manager: %s", reason)
                    return

            result = await self.execute_callback(sig)
            if not result:
                return

            order_id = result.get("order_id", "")
            if not order_id:
                return

            # Register pending order — portfolio update happens in handle_fill
            self._register_pending(sig, result)
            self._update_reserved_cash()

            # Tell strategy to track this order (adds to _active_orders for lifecycle)
            runner = self.strategies.get(sig.strategy)
            if runner and hasattr(runner.strategy, "track_order"):
                runner.strategy.track_order(
                    order_id=order_id,
                    token_id=result.get("token_id", sig.metadata.get("token_id", sig.market_id)),
                    side=sig.direction,
                    price=sig.price,
                    size=sig.metadata.get("shares", sig.size_usd / max(sig.price, 0.01)),
                    size_usd=sig.size_usd,
                )
        else:
            logger.warning("No executor set, skipping signal: %s", sig)

    async def _paper_execute(self, sig: Signal):
        """Simulate execution in paper mode (instant fill at signal price)."""

        # Drawdown guard — paper mode bypasses check_can_trade(), enforce here
        dd_pct = self.risk_manager._current_drawdown_pct()
        if dd_pct >= self.risk_manager.config.max_daily_drawdown_pct:
            if not self.risk_manager.killed:
                await self.risk_manager._trigger_kill_switch(
                    f"Daily drawdown {dd_pct:.1f}% >= {self.risk_manager.config.max_daily_drawdown_pct}%"
                )
            return

        # For Polymarket use token_id as position key so each token tracks separately
        position_market_id = sig.metadata.get("token_id", sig.market_id)

        # Cap BUY size to available cash — multiple signals in one cycle share the
        # same cash snapshot at generation time, so later signals can overdraw.
        size_usd = sig.size_usd
        if sig.direction == "buy":
            available = self.portfolio.available_cash
            if available < 1.0:
                logger.debug("Skipping paper BUY: insufficient cash ($%.2f)", available)
                return
            max_size = available / (1 + self._paper_fee_rate)
            size_usd = min(size_usd, max_size)
            if size_usd < 1.0:
                return

            # Total exposure cap — check with the cash-capped size
            max_exposure = self.portfolio.initial_capital * self.risk_manager.config.max_total_exposure_pct / 100
            if self.portfolio.total_exposure + size_usd > max_exposure:
                logger.info(
                    "Paper BUY skipped: exposure cap ($%.0f + $%.0f > $%.0f)",
                    self.portfolio.total_exposure, size_usd, max_exposure,
                )
                return

        orig_fee = sig.metadata.get("fee")
        if orig_fee is not None and sig.size_usd > 0:
            market_fee = orig_fee * (size_usd / sig.size_usd)
            # In paper mode use paper_fee_rate as floor (market fee may be 0% maker)
            fee = max(market_fee, size_usd * self._paper_fee_rate)
        else:
            fee = size_usd * self._paper_fee_rate

        trade = Trade(
            trade_id=str(uuid.uuid4())[:8],
            platform=Platform.POLYMARKET,  # default, overridden by strategy metadata
            market_id=position_market_id,
            symbol=sig.symbol,
            side=Side.BUY if sig.direction == "buy" else Side.SELL,
            price=sig.price,
            size=size_usd,
            fee=fee,
            slippage=0.0,
            strategy=sig.strategy,
            timestamp=time.time(),
            latency_ms=0.0,
        )
        if sig.metadata.get("platform"):
            trade.platform = Platform(sig.metadata["platform"])

        pnl = 0.0
        should_close = sig.metadata.get("close", False)

        # Auto-detect close: if existing position has opposite side, close it
        if not should_close:
            pos_key = self.portfolio._position_key(trade.platform, trade.market_id, trade.strategy)
            existing = self.portfolio.positions.get(pos_key)
            if existing and existing.side != trade.side:
                should_close = True
            elif trade.side == Side.SELL and not existing:
                # Paper mode: never open short positions — portfolio may have
                # dust-cleaned the position while MM inventory still shows shares.
                logger.debug("Paper SELL skipped: no position for %s", trade.market_id[:20])
                return

        if should_close:
            pnl = await self.portfolio.close_position(trade)
        else:
            await self.portfolio.open_position(trade)

        await self.risk_manager.record_trade(trade, filled=True, pnl=pnl)
        await self._notify_trade(trade, pnl)
        self._notify_strategy_fill(sig, trade)

        if self.db_callback:
            await self.db_callback(trade, pnl)
        await self._save_portfolio_state()

    async def _save_portfolio_state(self):
        """Persist portfolio state to DB after each trade."""
        if self._db:
            try:
                await self._db.save_state("portfolio", self.portfolio.to_dict())
            except Exception:
                logger.exception("Failed to save portfolio state")

    def _notify_strategy_fill(self, sig: Signal, trade):
        """Notify the originating strategy about a fill (for inventory tracking)."""
        runner = self.strategies.get(sig.strategy)
        if runner and hasattr(runner.strategy, "track_order"):
            runner.strategy.track_order(
                order_id=trade.trade_id,
                token_id=sig.metadata.get("token_id", sig.market_id),
                side=sig.direction,
                price=trade.price,
                size=sig.metadata.get("shares", trade.size / max(trade.price, 0.01)),
                size_usd=trade.size,
            )
        # In paper mode: simulate immediate fill.
        # In live mode: on_fill must be called by the user WebSocket handler when a
        # confirmed fill arrives — NOT here, to avoid treating a posted order as filled.
        if self.mode == "paper" and runner and hasattr(runner.strategy, "on_fill"):
            shares = sig.metadata.get("shares", trade.size / max(trade.price, 0.01))
            runner.strategy.on_fill(trade.trade_id, shares, trade.price)

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

    # --- Live mode: pending orders and fill reconciliation ---

    def _update_reserved_cash(self):
        """Recalculate cash reserved by pending BUY orders."""
        reserved = 0.0
        for p in self._pending_orders.values():
            if p["side"] == "buy":
                filled_usd = p.get("filled_so_far", 0.0) * p["price"]
                reserved += max(0, p["size_usd"] - filled_usd)
        self.portfolio.reserved_cash = reserved

    def _register_pending(self, sig: Signal, order_result: dict):
        """Track a live order as pending until fill arrives via user WS."""
        order_id = order_result.get("order_id", "")
        if not order_id:
            return
        self._pending_orders[order_id] = {
            "sig": sig,
            "order_id": order_id,
            "token_id": order_result.get("token_id", sig.metadata.get("token_id", sig.market_id)),
            "side": sig.direction,
            "price": sig.price,
            "size_usd": sig.size_usd,
            "placed_at": time.time(),
        }
        logger.info("Pending order registered: %s %s $%.0f @ %.4f (id: %s)",
                     sig.direction, sig.symbol, sig.size_usd, sig.price, order_id[:12])

    async def handle_fill(self, fill_data: dict):
        """Called by user WS when a trade/fill event arrives.

        Matches against pending orders, creates Trade, updates portfolio.
        Serialized via _fill_lock to prevent race conditions.
        """
        async with self._fill_lock:
            await self._handle_fill_inner(fill_data)

    async def _handle_fill_inner(self, fill_data: dict):
        event_type = fill_data.get("event_type", "")

        # Handle order cancellations/rejections — clean up pending immediately
        if event_type == "order":
            status = fill_data.get("status", "")
            if status in ("CANCELED", "CANCELLED", "REJECTED", "EXPIRED"):
                # Polymarket WS uses "id" for order events, not "order_id"
                order_id = fill_data.get("id") or fill_data.get("order_id", "")
                if order_id and order_id in self._pending_orders:
                    del self._pending_orders[order_id]
                    self._update_reserved_cash()
                    logger.info("Pending order removed (%s): %s", status, order_id[:12])
                elif order_id:
                    logger.debug("Cancel event for unknown order: %s (keys: %s)",
                                 order_id[:12], list(fill_data.keys()))
            return

        # Only process trade events
        if event_type != "trade":
            return

        # Only process MATCHED status — MINED/CONFIRMED are duplicate events
        # for the same fill as it progresses through the blockchain lifecycle.
        status = fill_data.get("status", "")
        if status != "MATCHED":
            return

        # Extract order IDs from the fill — check maker_orders and taker_order_id
        matched_orders = []
        for maker in fill_data.get("maker_orders", []):
            mid = maker.get("order_id", "")
            if mid in self._pending_orders:
                matched_orders.append((mid, maker))

        taker_id = fill_data.get("taker_order_id", "")
        if taker_id in self._pending_orders:
            matched_orders.append((taker_id, {
                "order_id": taker_id,
                "matched_amount": fill_data.get("size", "0"),
                "price": fill_data.get("price", "0"),
            }))

        if not matched_orders:
            # Log unmatched fills — likely our order whose pending expired during WS downtime
            maker_ids = [m.get("order_id", "?")[:12] for m in fill_data.get("maker_orders", [])]
            logger.warning(
                "Fill MATCHED but no pending order found (cleanup timeout?). "
                "taker=%s makers=%s asset=%s price=%s size=%s | pending_keys=%d",
                taker_id[:12] if taker_id else "none",
                maker_ids,
                fill_data.get("asset_id", "?")[:12],
                fill_data.get("price", "?"),
                fill_data.get("size", "?"),
                len(self._pending_orders),
            )
            if self.notify_callback:
                await self.notify_callback(
                    f"⚠️ Fill received but unmatched (WS downtime?): "
                    f"price={fill_data.get('price', '?')} "
                    f"size={fill_data.get('size', '?')}"
                )
            return

        for order_id, fill_info in matched_orders:
            pending = self._pending_orders.get(order_id)
            if not pending:
                continue
            sig = pending["sig"]
            filled_shares = float(fill_info.get("matched_amount", 0))
            filled_price = float(fill_info.get("price", pending["price"]))
            filled_usd = filled_shares * filled_price

            if filled_shares <= 0:
                continue

            # Track cumulative fill for partial fill support
            pending.setdefault("filled_so_far", 0.0)
            pending["filled_so_far"] += filled_shares
            original_shares = pending["size_usd"] / max(pending["price"], 0.01)
            if pending["filled_so_far"] >= original_shares * 0.99:
                del self._pending_orders[order_id]
            self._update_reserved_cash()

            trade = Trade(
                trade_id=f"{order_id}_{int(time.time())}",
                platform=Platform.POLYMARKET,
                market_id=pending["token_id"],
                symbol=sig.symbol,
                side=Side.BUY if sig.direction == "buy" else Side.SELL,
                price=filled_price,
                size=filled_usd,
                fee=sig.metadata.get("fee", 0.0) * (filled_usd / max(sig.size_usd, 0.01)),
                slippage=abs(filled_price - sig.price),
                strategy=sig.strategy,
                timestamp=time.time(),
                latency_ms=(time.time() - pending["placed_at"]) * 1000,
            )

            # Update portfolio
            pnl = 0.0
            should_close = sig.metadata.get("close", False)
            if not should_close:
                pos_key = self.portfolio._position_key(trade.platform, trade.market_id, trade.strategy)
                existing = self.portfolio.positions.get(pos_key)
                if existing and existing.side != trade.side:
                    should_close = True

            if should_close:
                pnl = await self.portfolio.close_position(trade)
            else:
                await self.portfolio.open_position(trade)

            # Notify strategy (update MM inventory)
            runner = self.strategies.get(sig.strategy)
            if runner and hasattr(runner.strategy, "on_fill"):
                runner.strategy.on_fill(order_id, filled_shares, filled_price)

            await self.risk_manager.record_trade(trade, filled=True, pnl=pnl)
            await self._notify_trade(trade, pnl)
            if self.db_callback:
                await self.db_callback(trade, pnl)
            await self._save_portfolio_state()

            logger.info("Fill reconciled: %s %s $%.2f @ %.4f (order: %s, filled: %.1f/%.1f)",
                        sig.direction, sig.symbol, filled_usd, filled_price, order_id[:12],
                        pending.get("filled_so_far", filled_shares), original_shares)

    async def _daily_reset_loop(self):
        """Reset daily risk counters at midnight UTC."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)
                next_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
                if now >= next_midnight:
                    from datetime import timedelta
                    next_midnight += timedelta(days=1)
                seconds_until = (next_midnight - now).total_seconds()
                await asyncio.sleep(seconds_until)
                self.risk_manager.reset_daily()
                logger.info("Daily risk reset completed")
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in daily reset loop")
                await asyncio.sleep(60)

    async def _heartbeat_loop(self):
        """Send heartbeat to Telegram every 30 min."""
        while self._running:
            try:
                await asyncio.sleep(1800)
                if self.notify_callback:
                    summary = self.portfolio.summary()
                    await self.notify_callback(
                        f"💓 Heartbeat | Equity: ${summary['equity']:.0f} "
                        f"(PnL: ${summary['equity_pnl']:+.2f}) "
                        f"| Exposure: {summary['exposure_pct']:.0f}% "
                        f"| Positions: {summary['open_positions']}"
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in heartbeat loop")

    async def _pending_order_cleanup_loop(self):
        """Remove stale pending orders that never filled (live mode)."""
        while self._running:
            try:
                await asyncio.sleep(60)
                now = time.time()
                stale = [oid for oid, p in self._pending_orders.items()
                         if now - p["placed_at"] > 360]  # 6 min (MM TTL 300s + buffer)
                for oid in stale:
                    del self._pending_orders[oid]
                    logger.warning("Pending order expired (no fill after 6min): %s", oid[:12])
                if stale:
                    self._update_reserved_cash()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in pending order cleanup")

    async def _reconciliation_loop(self):
        """Reconcile bot state vs exchange every 10 minutes (live mode only)."""
        await asyncio.sleep(120)  # let bot settle after startup
        while self._running:
            try:
                await asyncio.sleep(600)  # 10 min
                if self.mode != "live" or not self.poly_client:
                    continue
                await self._run_reconciliation()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in reconciliation loop")

    async def _run_reconciliation(self):
        """Compare local state vs exchange and alert on discrepancies."""
        alerts = []

        # 1. USDC balance check
        exchange_bal = await self.poly_client.get_exchange_balance()
        if exchange_bal is not None:
            local_cash = self.portfolio.available_cash
            delta = abs(exchange_bal - local_cash)
            if delta > 30.0:
                alerts.append(
                    f"Cash desync: local=${local_cash:.2f} vs exchange=${exchange_bal:.2f} (delta ${delta:.2f})"
                )
            logger.info("Recon cash: local=$%.2f exchange=$%.2f delta=$%.2f",
                        local_cash, exchange_bal, delta)

        # 2. Open orders check
        exchange_orders = await self.poly_client.get_exchange_orders()
        if exchange_orders is not None:
            local_count = len(self._pending_orders)
            exchange_count = len(exchange_orders)
            if abs(local_count - exchange_count) > 2:
                alerts.append(
                    f"Order desync: local={local_count} vs exchange={exchange_count}"
                )
            logger.info("Recon orders: local=%d exchange=%d", local_count, exchange_count)

        # 3. Position check (data API)
        exchange_positions = await self.poly_client.get_exchange_positions()
        if exchange_positions is not None:
            # Build exchange token set with non-zero size
            exchange_tokens = {}
            for ep in exchange_positions:
                asset = ep.get("asset", ep.get("token_id", ""))
                size = float(ep.get("size", 0))
                if size > 0.1:
                    exchange_tokens[asset] = size

            # Local positions (polymarket only)
            local_tokens = {}
            for pos in self.portfolio.positions.values():
                if pos.platform == Platform.POLYMARKET:
                    shares = pos.size / max(pos.avg_price, 0.01)
                    if shares > 0.1:
                        local_tokens[pos.market_id] = shares

            only_exchange = set(exchange_tokens) - set(local_tokens)
            only_local = set(local_tokens) - set(exchange_tokens)
            if only_exchange:
                alerts.append(
                    f"Positions on exchange not in bot: {len(only_exchange)}"
                )
            if only_local:
                alerts.append(
                    f"Positions in bot not on exchange: {len(only_local)}"
                )
            logger.info(
                "Recon positions: local=%d exchange=%d only_exchange=%d only_local=%d",
                len(local_tokens), len(exchange_tokens),
                len(only_exchange), len(only_local),
            )

        # Send Telegram alert only if alerts CHANGED since last run (suppress repeats)
        if alerts != self._last_recon_alerts and alerts and self.notify_callback:
            msg = "⚠️ Reconciliation alert:\n" + "\n".join(f"- {a}" for a in alerts)
            await self.notify_callback(msg)
        elif not alerts and self._last_recon_alerts:
            # Was alerting, now resolved
            if self.notify_callback:
                await self.notify_callback("✅ Reconciliation OK — discrepancies resolved")
        elif not alerts:
            logger.info("Reconciliation OK — no discrepancies")
        self._last_recon_alerts = alerts

    async def _shutdown(self):
        logger.info("Engine shutting down...")
        self._running = False

        # Clear pending orders — strategies' on_stop() cancels live orders on exchange
        if self._pending_orders:
            logger.info("Clearing %d pending orders on shutdown", len(self._pending_orders))
            self._pending_orders.clear()
            self._update_reserved_cash()

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

    # --- Paper reset ---

    async def reset_paper(self):
        """Reset paper portfolio to initial capital. Paper mode only."""
        if self.mode != "paper":
            raise RuntimeError("reset_paper() called in non-paper mode")
        p = self.portfolio
        async with p._lock:
            p.cash = p.initial_capital
            p.positions.clear()
            p.closed_trades.clear()
            p.realized_pnl = 0.0
            p.total_fees = 0.0
            p.total_llm_cost = 0.0
        self.risk_manager.daily_start_equity = p.equity
        if self._db:
            await self._db.save_state("portfolio", p.to_dict())
        logger.info("Paper portfolio reset. Cash: $%.2f", p.cash)

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
