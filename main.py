"""Trading Bot — Entry point.

Usage:
    python main.py              # Uses config/settings.yaml (paper mode)
    python main.py --live       # Override to live mode
"""
from __future__ import annotations

import asyncio
import logging
import sys

from connectors.ib_client import IBClient
from connectors.polymarket_client import PolymarketClient
from connectors.telegram_bot import TelegramBot
from core.engine import Engine
from core.portfolio import Portfolio
from core.risk_manager import RiskConfig, RiskManager
from data.db import Database
from strategies.ib_pairs import IBPairsTrader
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
    risk_config = RiskConfig.from_dict(config.get("risk", {}))
    risk_manager = RiskManager(portfolio, risk_config)
    engine = Engine(portfolio, risk_manager, config)

    # --- Database ---
    db = Database()
    await db.connect()
    engine.set_db(db.log_trade)

    # --- Telegram ---
    tg = TelegramBot(config.get("telegram", {}), engine=engine)
    await tg.start()
    engine.set_notify(tg.send_message)

    # --- Polymarket connector ---
    poly_client = PolymarketClient(config.get("polymarket", {}))
    await poly_client.connect()

    # --- Strategies ---
    mm = PolyMarketMaker(
        name="poly_mm",
        portfolio=portfolio,
        risk_manager=risk_manager,
        config=config.get("polymarket", {}),
        poly_client=poly_client,
    )
    engine.add_strategy(mm, interval_seconds=5.0)

    # --- IB connector + pairs trading (paper only in Phase 1) ---
    ib_client = IBClient(config.get("ib", {}))
    await ib_client.connect()

    pairs_trader = IBPairsTrader(
        name="ib_pairs",
        portfolio=portfolio,
        risk_manager=risk_manager,
        config=config.get("ib", {}),
        ib_client=ib_client,
    )
    engine.add_strategy(pairs_trader, interval_seconds=30.0)

    logger.info("Starting bot in %s mode with $%.0f capital", mode, capital)

    try:
        await engine.run()
    finally:
        # Graceful shutdown of all components
        logger.info("Shutting down components...")
        await ib_client.disconnect()
        await poly_client.disconnect()
        await tg.stop()
        await db.disconnect()
        logger.info("All components stopped")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
