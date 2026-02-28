"""Telegram bot — notifications + commands for monitoring and control."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from telegram import Update
from telegram.error import RetryAfter
from telegram.ext import Application, CommandHandler, ContextTypes

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot for trade notifications and bot control.

    Commands: /status, /trades, /stop, /start, /risk, /pnl
    """

    def __init__(self, config: dict, engine=None):
        self.token = config.get("bot_token", "")
        self.chat_id = str(config.get("chat_id", ""))
        self.engine = engine  # set after engine creation

        self._app: Optional[Application] = None
        self._running = False
        self._reset_pending_at: Optional[float] = None  # timestamp of pending reset
        self._flood_blocked_until: float = 0.0  # Telegram RetryAfter blackout
        self._last_notify_at: float = 0.0       # for notification throttle
        self._notify_min_interval: float = 10.0 # seconds between trade notifications

    def set_engine(self, engine):
        self.engine = engine

    async def start(self):
        if not self.token or self.token.startswith("${"):
            logger.warning("Telegram bot token not configured, running without notifications")
            return

        self._app = Application.builder().token(self.token).build()

        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("trades", self._cmd_trades))
        self._app.add_handler(CommandHandler("stop", self._cmd_stop))
        self._app.add_handler(CommandHandler("start", self._cmd_start))
        self._app.add_handler(CommandHandler("risk", self._cmd_risk))
        self._app.add_handler(CommandHandler("pnl", self._cmd_pnl))
        self._app.add_handler(CommandHandler("reset", self._cmd_reset))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        self._running = True
        logger.info("Telegram bot started")

    async def stop(self):
        if self._app and self._running:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            self._running = False
            logger.info("Telegram bot stopped")

    async def send_message(self, text: str, force: bool = False):
        """Send a message to the configured chat. Used as notify callback.

        Handles Telegram flood control (RetryAfter) by honouring the blackout
        window. Trade notifications are also throttled to avoid future floods;
        pass force=True to bypass the throttle (used for kill-switch alerts).
        """
        if not self._app or not self.chat_id:
            return

        now = time.time()

        # Honor active RetryAfter blackout
        if now < self._flood_blocked_until:
            remaining = self._flood_blocked_until - now
            logger.debug("Telegram blocked for %.0fs, dropping message", remaining)
            return

        # Throttle high-frequency trade notifications
        if not force and now - self._last_notify_at < self._notify_min_interval:
            return

        try:
            await self._app.bot.send_message(chat_id=self.chat_id, text=text)
            self._last_notify_at = now
        except RetryAfter as e:
            self._flood_blocked_until = now + e.retry_after
            logger.warning("Telegram flood control: blocked for %ds", e.retry_after)
        except Exception:
            logger.exception("Failed to send Telegram message")

    # --- Commands ---

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        if not self.engine:
            await update.message.reply_text("Engine not connected")
            return

        s = self.engine.portfolio.summary()
        text = (
            f"<b>Status ({self.engine.mode})</b>\n"
            f"Equity: ${s['equity']:.2f} (PnL: ${s['equity_pnl']:+.2f})\n"
            f"Cash: ${s['cash']:.2f}\n"
            f"Exposure: ${s['total_exposure']:.2f} ({s['exposure_pct']:.0f}%)\n"
            f"PnL: ${s['net_pnl']:+.2f}\n"
            f"Drawdown: {s['drawdown_pct']:.1f}%\n"
            f"Positions: {s['open_positions']}\n"
            f"Fees: ${s['total_fees']:.2f} | LLM: ${s['llm_cost']:.2f}"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        if not self.engine:
            await update.message.reply_text("Engine not connected")
            return

        trades = list(self.engine.portfolio.closed_trades)[-10:]
        if not trades:
            await update.message.reply_text("No trades yet")
            return

        lines = ["<b>Last trades:</b>"]
        for t in reversed(trades):
            lines.append(
                f"{'B' if t.side.value == 'buy' else 'S'} "
                f"${t.size:.0f} {t.symbol} @ {t.price:.4f} "
                f"[{t.strategy}]"
            )
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def _cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        if self.engine:
            await self.engine.risk_manager.manual_kill()
            await update.message.reply_text("Kill switch activated. Use /start to resume.")

    async def _cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        if self.engine:
            self.engine.risk_manager.resume()
            await update.message.reply_text("Trading resumed.")

    async def _cmd_risk(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        if not self.engine:
            await update.message.reply_text("Engine not connected")
            return

        r = self.engine.risk_manager.status()
        lines = [
            f"<b>Risk Status</b>",
            f"Kill switch: {'ACTIVE' if r['killed'] else 'off'}",
            f"Daily drawdown: {r['daily_drawdown_pct']:.2f}%",
        ]
        if r["paused_strategies"]:
            for strat, secs in r["paused_strategies"].items():
                lines.append(f"Paused: {strat} ({secs:.0f}s left)")
        for strat, metrics in r.get("execution_metrics", {}).items():
            lines.append(
                f"{strat}: fill={metrics['fill_rate']:.0%} "
                f"slip={metrics['avg_slippage']:.2%} "
                f"lat={metrics['avg_latency_ms']:.0f}ms"
            )
        await update.message.reply_text("\n".join(lines), parse_mode="HTML")

    async def _cmd_pnl(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        if not self.engine:
            await update.message.reply_text("Engine not connected")
            return

        s = self.engine.portfolio.summary()
        text = (
            f"<b>PnL Summary</b>\n"
            f"Realized: ${s['realized_pnl']:+.2f}\n"
            f"Unrealized: ${s['unrealized_pnl']:+.2f}\n"
            f"Fees: -${s['total_fees']:.2f}\n"
            f"LLM cost: -${s['llm_cost']:.2f}\n"
            f"<b>Net: ${s['net_pnl']:+.2f}</b>"
        )
        await update.message.reply_text(text, parse_mode="HTML")

    async def _cmd_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self._is_authorized(update):
            return
        if not self.engine:
            await update.message.reply_text("Engine not connected")
            return
        if self.engine.mode != "paper":
            await update.message.reply_text("Reset only available in paper mode.")
            return

        import time
        now = time.time()
        CONFIRM_WINDOW = 30  # seconds to confirm

        if self._reset_pending_at and now - self._reset_pending_at < CONFIRM_WINDOW:
            # Second /reset within window → execute
            self._reset_pending_at = None
            before = self.engine.portfolio.summary()
            await self.engine.reset_paper()
            after = self.engine.portfolio.summary()
            await update.message.reply_text(
                f"✅ <b>Paper portfolio reset</b>\n"
                f"Before: equity ${before['equity']:.2f}, {before['open_positions']} positions\n"
                f"After:  equity ${after['equity']:.2f}, cash ${after['cash']:.2f}",
                parse_mode="HTML",
            )
        else:
            # First /reset → show warning and arm the confirmation
            self._reset_pending_at = now
            s = self.engine.portfolio.summary()
            await update.message.reply_text(
                f"⚠️ <b>Reset paper portfolio?</b>\n"
                f"Current equity: ${s['equity']:.2f} (PnL: ${s['equity_pnl']:+.2f})\n"
                f"Positions: {s['open_positions']} | Fees: ${s['total_fees']:.2f}\n\n"
                f"Cash will reset to ${self.engine.portfolio.initial_capital:.0f}, "
                f"all positions and PnL counters cleared.\n\n"
                f"Send /reset again within {CONFIRM_WINDOW}s to confirm.",
                parse_mode="HTML",
            )

    def _is_authorized(self, update: Update) -> bool:
        """Only respond to the configured chat_id."""
        return str(update.effective_chat.id) == self.chat_id
