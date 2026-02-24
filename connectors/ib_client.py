"""Interactive Brokers connector via ib_async (successor to ib_insync).

Provides async connection with auto-reconnect, historical data,
real-time streaming, and order placement for pairs trading.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

from ib_async import IB, Contract, LimitOrder, MarketOrder, Stock, util

logger = logging.getLogger(__name__)


@dataclass
class IBTicker:
    """Simplified ticker snapshot for strategy consumption."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: float


class IBClient:
    """Async IB connector with auto-reconnect and streaming.

    Usage:
        client = IBClient(config)
        await client.connect()
        bars = await client.get_historical("AAPL", days=120)
        await client.subscribe(["AAPL", "MSFT"], callback)
        await client.place_order("AAPL", "BUY", 10, order_type="MKT")
    """

    def __init__(self, config: dict):
        self._host = config.get("host", "127.0.0.1")
        self._port = config.get("port", 4002)
        self._client_id = config.get("client_id", 1)

        self.ib = IB()
        self._connected = False
        self._reconnecting = False
        self._subscriptions: dict[str, Contract] = {}
        self._tickers: dict[str, IBTicker] = {}
        self._on_tick: Optional[Callable] = None
        self._event_registered = False

        # Attach events once
        self.ib.connectedEvent += self._on_connected
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.errorEvent += self._on_error

    # --- Connection ---

    async def connect(self):
        try:
            await self.ib.connectAsync(
                self._host, self._port, self._client_id, timeout=10,
            )
            logger.info(
                "Connected to IB at %s:%d (clientId=%d)",
                self._host, self._port, self._client_id,
            )
        except (ConnectionRefusedError, OSError, asyncio.TimeoutError) as e:
            logger.warning("IB connection failed: %s — running without IB", e)

    async def disconnect(self):
        self._connected = False
        for symbol in list(self._subscriptions):
            self._cancel_mkt_data(symbol)
        if self.ib.isConnected():
            self.ib.disconnect()
        logger.info("IB client disconnected")

    def _on_connected(self):
        self._connected = True
        self._reconnecting = False
        logger.info("IB connected event fired")

    def _on_disconnected(self):
        was_connected = self._connected
        self._connected = False
        if was_connected and not self._reconnecting:
            logger.warning("IB disconnected unexpectedly, scheduling reconnect")
            asyncio.ensure_future(self._reconnect())

    def _on_error(self, reqId, errorCode, errorString, contract):
        # Filter informational messages (codes 2000+)
        if errorCode >= 2000:
            logger.debug("IB info %d: %s", errorCode, errorString)
        else:
            logger.error("IB error %d (req %d): %s", errorCode, reqId, errorString)

    async def _reconnect(self):
        self._reconnecting = True
        delay = 5
        max_delay = 120
        while not self.ib.isConnected():
            try:
                logger.info("Attempting IB reconnection in %ds...", delay)
                await asyncio.sleep(delay)
                await self.ib.connectAsync(
                    self._host, self._port, self._client_id, timeout=10,
                )
                if self.ib.isConnected():
                    logger.info("IB reconnected successfully")
                    # Re-subscribe to market data
                    await self._resubscribe()
                    break
            except (ConnectionRefusedError, OSError, asyncio.TimeoutError):
                delay = min(delay * 2, max_delay)
        self._reconnecting = False

    async def _resubscribe(self):
        """Re-subscribe to market data after reconnection."""
        symbols = list(self._subscriptions.keys())
        self._subscriptions.clear()
        for symbol in symbols:
            contract = self._make_contract(symbol)
            self._subscriptions[symbol] = contract
            self.ib.reqMktData(contract, "", False, False)
        if symbols:
            logger.info("Re-subscribed to %d symbols", len(symbols))

    # --- Market Data ---

    async def get_historical(
        self, symbol: str, days: int = 120, bar_size: str = "1 day",
    ) -> list[dict]:
        """Fetch historical daily bars. Returns list of dicts with OHLCV."""
        if not self.ib.isConnected():
            logger.warning("IB not connected, cannot fetch historical data")
            return []

        contract = self._make_contract(symbol)
        try:
            bars = await asyncio.wait_for(
                self.ib.reqHistoricalDataAsync(
                    contract,
                    endDateTime="",
                    durationStr=f"{days} D",
                    barSizeSetting=bar_size,
                    whatToShow="ADJUSTED_LAST",
                    useRTH=True,
                ),
                timeout=30,
            )
        except asyncio.TimeoutError:
            logger.error("Historical data timeout for %s", symbol)
            return []

        if not bars:
            logger.warning("No historical data for %s", symbol)
            return []

        return [
            {
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
            for bar in bars
        ]

    async def subscribe(self, symbols: list[str], on_tick: Callable = None):
        """Subscribe to real-time Level 1 quotes for given symbols."""
        if not self.ib.isConnected():
            logger.warning("IB not connected, cannot subscribe")
            return

        if on_tick:
            self._on_tick = on_tick
        # Register event handler only once
        if not self._event_registered:
            self.ib.pendingTickersEvent += self._handle_tickers
            self._event_registered = True

        for symbol in symbols:
            if symbol in self._subscriptions:
                continue
            contract = self._make_contract(symbol)
            self._subscriptions[symbol] = contract
            self.ib.reqMktData(contract, "", False, False)

        logger.info("Subscribed to %d IB symbols", len(symbols))

    def _cancel_mkt_data(self, symbol: str):
        contract = self._subscriptions.pop(symbol, None)
        if contract and self.ib.isConnected():
            self.ib.cancelMktData(contract)

    def _handle_tickers(self, tickers):
        """Process incoming ticker updates from IB."""
        for t in tickers:
            symbol = t.contract.symbol
            self._tickers[symbol] = IBTicker(
                symbol=symbol,
                bid=t.bid if t.bid and t.bid > 0 else 0.0,
                ask=t.ask if t.ask and t.ask > 0 else 0.0,
                last=t.last if t.last and t.last > 0 else 0.0,
                volume=t.volume if t.volume else 0.0,
                timestamp=time.time(),
            )
        if self._on_tick:
            updated_symbols = [t.contract.symbol for t in tickers]
            try:
                self._on_tick(updated_symbols)
            except Exception:
                logger.exception("Error in tick callback")

    def get_ticker(self, symbol: str) -> Optional[IBTicker]:
        return self._tickers.get(symbol)

    def get_mid_price(self, symbol: str) -> float:
        """Get mid price for a symbol. Returns 0.0 if unavailable."""
        ticker = self._tickers.get(symbol)
        if not ticker:
            return 0.0
        if ticker.bid > 0 and ticker.ask > 0:
            return (ticker.bid + ticker.ask) / 2
        if ticker.last > 0:
            return ticker.last
        return 0.0

    # --- Order Execution ---

    async def place_order(
        self, symbol: str, action: str, quantity: int,
        order_type: str = "MKT", limit_price: float = 0.0,
    ) -> Optional[dict]:
        """Place an order. Returns fill info dict or None on failure.

        Args:
            action: 'BUY' or 'SELL'
            order_type: 'MKT' or 'LMT'
        """
        if not self.ib.isConnected():
            logger.error("IB not connected, cannot place order")
            return None

        contract = self._make_contract(symbol)

        if order_type == "LMT" and limit_price > 0:
            order = LimitOrder(action, quantity, limit_price)
        else:
            order = MarketOrder(action, quantity)

        trade = self.ib.placeOrder(contract, order)
        logger.info("Placed %s %s %d %s", order_type, action, quantity, symbol)

        # Wait for fill (up to 30 seconds for market orders)
        timeout = 30 if order_type == "MKT" else 5
        start = time.time()
        while not trade.isDone() and (time.time() - start) < timeout:
            await asyncio.sleep(0.5)

        if trade.orderStatus.status == "Filled":
            return {
                "symbol": symbol,
                "action": action,
                "quantity": trade.orderStatus.filled,
                "avg_price": trade.orderStatus.avgFillPrice,
                "order_id": trade.order.orderId,
                "status": "Filled",
                "latency_ms": (time.time() - start) * 1000,
            }

        status = trade.orderStatus.status
        if status in ("Cancelled", "ApiCancelled", "Inactive"):
            logger.warning("Order %s: %s %d %s", status, action, quantity, symbol)
            return None

        # Still pending — return partial info
        logger.warning(
            "Order still %s after %ds: %s %d %s",
            status, timeout, action, quantity, symbol,
        )
        return {
            "symbol": symbol,
            "action": action,
            "quantity": trade.orderStatus.filled,
            "avg_price": trade.orderStatus.avgFillPrice,
            "order_id": trade.order.orderId,
            "status": status,
            "latency_ms": (time.time() - start) * 1000,
        }

    async def cancel_all_orders(self):
        """Cancel all open orders."""
        if not self.ib.isConnected():
            return
        open_trades = self.ib.openTrades()
        for trade in open_trades:
            self.ib.cancelOrder(trade.order)
        if open_trades:
            logger.info("Cancelled %d open orders", len(open_trades))

    # --- Account Info ---

    def get_positions(self) -> list[dict]:
        """Get current positions."""
        if not self.ib.isConnected():
            return []
        return [
            {
                "symbol": pos.contract.symbol,
                "position": pos.position,
                "avg_cost": pos.avgCost,
                "account": pos.account,
            }
            for pos in self.ib.positions()
        ]

    def get_account_value(self, tag: str = "NetLiquidation") -> float:
        """Get an account value by tag name."""
        if not self.ib.isConnected():
            return 0.0
        for av in self.ib.accountValues():
            if av.tag == tag and av.currency == "USD":
                try:
                    return float(av.value)
                except (ValueError, TypeError):
                    pass
        return 0.0

    # --- Helpers ---

    @staticmethod
    def _make_contract(symbol: str) -> Stock:
        return Stock(symbol, "SMART", "USD")

    @property
    def is_connected(self) -> bool:
        return self.ib.isConnected()
