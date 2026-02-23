"""Trading Bot — Entry point.

Usage:
    python main.py              # Uses config/settings.yaml
    python main.py --live       # Override to live mode
"""
from __future__ import annotations

import asyncio
import sys

from core.engine import Engine
from core.portfolio import Portfolio
from core.risk_manager import RiskConfig, RiskManager
from utils.helpers import load_config
from utils.logger import setup_logging


async def main():
    config = load_config()
    setup_logging(config.get("general", {}).get("log_level", "INFO"))

    # Override mode from CLI
    if "--live" in sys.argv:
        config["general"]["mode"] = "live"

    capital = config["general"]["capital_usd"]
    mode = config["general"]["mode"]

    portfolio = Portfolio(capital_usd=capital)
    risk_config = RiskConfig.from_dict(config.get("risk", {}))
    risk_manager = RiskManager(portfolio, risk_config)

    engine = Engine(portfolio, risk_manager, config)

    # --- Register strategies (add as they're implemented) ---
    # from strategies.poly_market_maker import PolyMarketMaker
    # mm = PolyMarketMaker("poly_mm", portfolio, risk_manager, config.get("polymarket", {}))
    # engine.add_strategy(mm, interval_seconds=5.0)

    # --- Register connectors (add as they're implemented) ---
    # from connectors.telegram_bot import TelegramBot
    # tg = TelegramBot(config.get("telegram", {}))
    # engine.set_notify(tg.send_message)

    import logging
    logger = logging.getLogger("main")
    logger.info("Starting bot in %s mode with $%.0f capital", mode, capital)
    logger.info("No strategies registered yet — implement and uncomment in main.py")

    await engine.run()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
