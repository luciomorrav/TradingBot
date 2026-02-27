"""ExecutionRouter — Routes trade signals to the correct platform connector.

Places orders on Polymarket (CLOB) or IB (ib_async) based on signal metadata.
Returns order response dicts — actual fills are reconciled via user WebSocket.
"""
from __future__ import annotations

import logging
from typing import Optional

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

    async def execute(self, sig: Signal) -> Optional[dict]:
        """Execute a signal on the appropriate platform.

        Returns order response dict with at least {order_id, status, token_id, side, price, size}.
        Portfolio update happens later when fill is confirmed via user WS.
        """
        platform = sig.metadata.get("platform", "polymarket")

        if platform == "polymarket":
            return await self._execute_polymarket(sig)
        elif platform == "ib":
            return await self._execute_ib(sig)
        else:
            logger.error("Unknown platform: %s", platform)
            return None

    async def _execute_polymarket(self, sig: Signal) -> Optional[dict]:
        """Place order on Polymarket CLOB. Returns order response dict.

        Portfolio update happens later when fill is confirmed via user WS (handle_fill).
        """
        client = self._executors.get("polymarket")
        if not client:
            logger.error("No Polymarket executor registered")
            return None

        try:
            side = "BUY" if sig.direction == "buy" else "SELL"
            shares = sig.size_usd / sig.price if sig.price > 0 else 0
            if shares < 1:
                logger.warning("Order too small: %.2f shares", shares)
                return None

            token_id = sig.metadata.get("token_id", sig.market_id)

            result = await client.place_order(
                token_id=token_id,
                side=side,
                price=sig.price,
                size=shares,
            )

            if not result or not result.get("order_id"):
                return None

            logger.info(
                "Polymarket order placed: %s %s $%.2f @ %.4f (id: %s) [%s]",
                sig.direction, sig.symbol, sig.size_usd, sig.price,
                result.get("order_id", "?")[:12], result.get("status", "?"),
            )
            return result

        except Exception:
            logger.exception("Polymarket execution failed for %s", sig)
            return None

    async def _execute_ib(self, sig: Signal) -> Optional[dict]:
        """Execute on Interactive Brokers. Returns order response dict."""
        client = self._executors.get("ib")
        if not client:
            logger.error("No IB executor registered")
            return None

        try:
            action = "BUY" if sig.direction == "buy" else "SELL"
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

            logger.info(
                "IB order placed: %s %d %s @ %.2f [%s]",
                action, quantity, sig.symbol, price, result.get("status"),
            )
            return result

        except Exception:
            logger.exception("IB execution failed for %s", sig)
            return None
