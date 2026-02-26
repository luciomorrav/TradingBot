"""ExecutionRouter — Routes trade signals to the correct platform connector.

Bridges engine signals to Polymarket (CLOB) or IB (ib_async) based on
signal metadata. Handles fill conversion to Trade objects for portfolio.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from core.portfolio import Platform, Side, Trade
from strategies.base_strategy import Signal

logger = logging.getLogger(__name__)


class ExecutionRouter:
    """Routes signals to platform-specific executors and returns Trade objects.

    Usage:
        router = ExecutionRouter()
        router.register("polymarket", poly_executor)
        router.register("ib", ib_executor)
        engine.set_executor(router.execute)
    """

    def __init__(self):
        self._executors: dict[str, object] = {}

    def register(self, platform: str, executor):
        """Register a platform executor.

        Polymarket executor: PolymarketClient (has place_order)
        IB executor: IBClient (has place_order)
        """
        self._executors[platform] = executor
        logger.info("Registered executor for platform: %s", platform)

    async def execute(self, sig: Signal) -> Optional[Trade]:
        """Execute a signal on the appropriate platform. Returns Trade or None."""
        platform = sig.metadata.get("platform", "polymarket")

        if platform == "polymarket":
            return await self._execute_polymarket(sig)
        elif platform == "ib":
            return await self._execute_ib(sig)
        else:
            logger.error("Unknown platform: %s", platform)
            return None

    async def _execute_polymarket(self, sig: Signal) -> Optional[Trade]:
        """Execute on Polymarket via CLOB API."""
        client = self._executors.get("polymarket")
        if not client:
            logger.error("No Polymarket executor registered")
            return None

        start = time.time()
        try:
            # Convert signal direction to CLOB side
            side = "BUY" if sig.direction == "buy" else "SELL"
            # Size in shares: size_usd / price
            shares = sig.size_usd / sig.price if sig.price > 0 else 0
            if shares < 1:
                logger.warning("Order too small: %.2f shares", shares)
                return None

            # Use token_id from metadata (market_id != token_id in Polymarket)
            token_id = sig.metadata.get("token_id", sig.market_id)

            result = await client.place_order(
                token_id=token_id,
                side=side,
                price=sig.price,
                size=shares,
            )

            if not result:
                return None

            latency = (time.time() - start) * 1000
            status = result.get("status", "")

            # WARN: treating order post as fill — real fills come via user WS
            if status not in ("paper",):
                logger.warning(
                    "Live order treated as immediate fill (no user WS reconciliation yet). "
                    "Order ID: %s", result.get("order_id", "?"),
                )

            # Determine fee from market data or default
            fee = sig.metadata.get("fee", 0.0)

            trade = Trade(
                trade_id=result.get("order_id", str(uuid.uuid4())[:8]),
                platform=Platform.POLYMARKET,
                market_id=sig.market_id,
                symbol=sig.symbol,
                side=Side.BUY if sig.direction == "buy" else Side.SELL,
                price=sig.price,
                size=sig.size_usd,
                fee=fee,
                slippage=0.0,  # TODO: calculate from fill price vs signal price
                strategy=sig.strategy,
                timestamp=time.time(),
                latency_ms=latency,
            )

            logger.info(
                "Polymarket fill: %s %s $%.2f @ %.4f (%.0fms) [%s]",
                sig.direction, sig.symbol, sig.size_usd, sig.price,
                latency, status,
            )
            return trade

        except Exception:
            logger.exception("Polymarket execution failed for %s", sig)
            return None

    async def _execute_ib(self, sig: Signal) -> Optional[Trade]:
        """Execute on Interactive Brokers."""
        client = self._executors.get("ib")
        if not client:
            logger.error("No IB executor registered")
            return None

        start = time.time()
        try:
            action = "BUY" if sig.direction == "buy" else "SELL"
            # IB uses share quantity, not USD notional
            price = sig.price
            quantity = max(1, int(sig.size_usd / price)) if price > 0 else 0
            if quantity < 1:
                logger.warning("IB order too small: %d shares", quantity)
                return None

            result = await client.place_order(
                symbol=sig.symbol,
                action=action,
                quantity=quantity,
                order_type="MKT",
            )

            if not result or result.get("status") in ("Cancelled", "ApiCancelled", "Inactive"):
                return None

            latency = result.get("latency_ms", (time.time() - start) * 1000)
            fill_price = result.get("avg_price", price)
            slippage = abs(fill_price - price) / price if price > 0 else 0

            trade = Trade(
                trade_id=str(result.get("order_id", uuid.uuid4()))[:8],
                platform=Platform.IB,
                market_id=sig.market_id,
                symbol=sig.symbol,
                side=Side.BUY if sig.direction == "buy" else Side.SELL,
                price=fill_price,
                size=result.get("quantity", quantity) * fill_price,
                fee=0.0,  # IB reports commissions via commissionReportEvent
                slippage=slippage,
                strategy=sig.strategy,
                timestamp=time.time(),
                latency_ms=latency,
            )

            logger.info(
                "IB fill: %s %d %s @ %.2f (slip=%.4f, %.0fms) [%s]",
                action, quantity, sig.symbol, fill_price,
                slippage, latency, result.get("status"),
            )
            return trade

        except Exception:
            logger.exception("IB execution failed for %s", sig)
            return None
