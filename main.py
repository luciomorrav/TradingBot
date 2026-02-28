"""Trading Bot — Entry point.

Usage:
    python main.py              # Uses config/settings.yaml (paper mode)
    python main.py --live       # Override to live mode
"""
from __future__ import annotations

import asyncio
import logging
import sys

from connectors.polymarket_client import PolymarketClient
from connectors.telegram_bot import TelegramBot
from core.engine import Engine
from core.execution_router import ExecutionRouter
from core.portfolio import Portfolio
from core.risk_manager import RiskConfig, RiskManager
from data.db import Database
from strategies.poly_market_maker import PolyMarketMaker
from utils.helpers import load_config
from utils.logger import setup_logging

logger = logging.getLogger("main")


async def main():
    config = load_config()
    setup_logging(config.get("general", {}).get("log_level", "INFO"))

    # Override mode from CLI
    if "--live" in sys.argv:
        config["general"]["mode"] = "live"

    capital = config["general"]["capital_usd"]
    mode = config["general"]["mode"]

    # --- Core ---
    portfolio = Portfolio(capital_usd=capital)

    # --- Database ---
    db = Database()
    await db.connect()

    # Restore portfolio state if available
    saved_portfolio = await db.load_state("portfolio")
    if saved_portfolio:
        portfolio.restore_from(saved_portfolio)
        logger.info(
            "Restored portfolio: %d positions, $%.2f cash, PnL $%.2f",
            len(portfolio.positions), portfolio.cash, portfolio.realized_pnl,
        )

    risk_config = RiskConfig.from_dict(config.get("risk", {}))
    risk_manager = RiskManager(portfolio, risk_config)
    engine = Engine(portfolio, risk_manager, config)
    engine.set_db(db.log_trade, db_instance=db)

    # --- Telegram ---
    tg = TelegramBot(config.get("telegram", {}), engine=engine)
    await tg.start()
    engine.set_notify(tg.send_message)

    # --- Polymarket connector ---
    poly_client = PolymarketClient(config.get("polymarket", {}))
    poly_client.set_notify(tg.send_message)  # WS disconnect/reconnect alerts
    await poly_client.connect()

    # --- Strategies ---
    mm = PolyMarketMaker(
        name="poly_mm",
        portfolio=portfolio,
        risk_manager=risk_manager,
        config=config.get("polymarket", {}),
        poly_client=poly_client,
    )
    mm._db = db
    engine.add_strategy(mm, interval_seconds=5.0)

    # --- IB connector + pairs trading (only if IB config is present and enabled) ---
    ib_client = None
    ib_config = config.get("ib", {})
    if ib_config.get("enabled", False):
        from connectors.ib_client import IBClient
        from strategies.ib_pairs import IBPairsTrader

        ib_client = IBClient(ib_config)
        await ib_client.connect()

        pairs_trader = IBPairsTrader(
            name="ib_pairs",
            portfolio=portfolio,
            risk_manager=risk_manager,
            config=ib_config,
            ib_client=ib_client,
        )
        engine.add_strategy(pairs_trader, interval_seconds=30.0)

    # --- Execution router (live mode only) ---
    if mode == "live":
        router = ExecutionRouter()
        router.register("polymarket", poly_client)
        if ib_client:
            router.register("ib", ib_client)
        engine.set_executor(router.execute)

        # Wire user WS for fill reconciliation (dedicated callback, not shared with market WS)
        poly_client.on_user_trade(engine.handle_fill)
        engine.set_ws_check(lambda: poly_client.user_ws_connected)
        await poly_client.subscribe_user()
        logger.info("Live execution router active with user WS")

    logger.info("Starting bot in %s mode with $%.0f capital", mode, capital)

    try:
        await engine.run()
    finally:
        # Graceful shutdown of all components
        logger.info("Shutting down components...")
        if ib_client:
            await ib_client.disconnect()
        await poly_client.disconnect()
        await tg.stop()
        await db.disconnect()
        logger.info("All components stopped")


if __name__ == "__main__":
    asyncio.run(main())
