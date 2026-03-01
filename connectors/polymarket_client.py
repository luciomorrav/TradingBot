"""Polymarket CLOB connector — WebSocket-first with REST for orders.

Market data flows via WebSocket (no rate limits).
Order placement via REST (60 orders/min limit).
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import aiohttp

logger = logging.getLogger(__name__)

CLOB_HOST = "https://clob.polymarket.com"
GAMMA_HOST = "https://gamma-api.polymarket.com"
WS_MARKET = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
WS_USER = "wss://ws-subscriptions-clob.polymarket.com/ws/user"


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    market_id: str
    token_id: str
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    timestamp: float = 0.0

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 1.0

    @property
    def mid_price(self) -> float:
        if not self.bids and not self.asks:
            return 0.5
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def bid_depth(self) -> float:
        return sum(lvl.size for lvl in self.bids)

    @property
    def ask_depth(self) -> float:
        return sum(lvl.size for lvl in self.asks)


@dataclass
class Market:
    id: str
    question: str
    slug: str
    active: bool
    end_date: str
    tokens: list[dict]  # [{token_id, outcome, price}]
    volume: float = 0.0
    liquidity: float = 0.0
    category: str = ""
    fee: float = 0.0


class PolymarketClient:
    """WebSocket-first Polymarket connector.

    Usage:
        client = PolymarketClient(config)
        await client.connect()
        await client.subscribe_market(["token_id_1", "token_id_2"])
        book = client.get_order_book("token_id_1")
    """

    def __init__(self, config: dict):
        self.config = config
        self._api_key = str(config.get("api_key", ""))
        self._api_secret = str(config.get("api_secret", ""))
        self._api_passphrase = str(config.get("api_passphrase", ""))
        # YAML parses 0x... as hex int, so force to string and restore 0x prefix
        pk = config.get("private_key", "")
        if isinstance(pk, int):
            pk = hex(pk)
        self._private_key = str(pk)
        self._chain_id = config.get("chain_id", 137)

        # py-clob-client for live order signing (initialized in connect() if creds present)
        self._clob = None
        self._live_enabled = False

        self._session: Optional[aiohttp.ClientSession] = None
        self._ws_market: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_user: Optional[aiohttp.ClientWebSocketResponse] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._user_ws_task: Optional[asyncio.Task] = None

        self._order_books: dict[str, OrderBook] = {}
        self._markets: dict[str, Market] = {}
        self._subscribed_tokens: set[str] = set()

        self._on_book_update: Optional[Callable] = None
        self._on_trade: Optional[Callable] = None
        self._on_user_trade: Optional[Callable] = None
        self._notify_callback: Optional[Callable] = None  # async notify(msg)

        self._running = False
        self._user_ws_connected = False  # fail-closed: block live orders if WS down

        # Rate limiter for live orders (Polymarket limit: 60/min)
        self._order_timestamps: list[float] = []
        self._order_rate_limit = 50  # stay under 60/min with margin

    # --- Lifecycle ---

    async def connect(self):
        timeout = aiohttp.ClientTimeout(total=30)
        self._session = aiohttp.ClientSession(timeout=timeout)
        self._running = True

        # Initialize py-clob-client for live order signing if credentials are present
        if self._private_key and not self._private_key.startswith("${"):
            try:
                from py_clob_client.client import ClobClient
                from py_clob_client.clob_types import ApiCreds

                # Polymarket proxy wallets use signature_type=1 (POLY_PROXY)
                # EOA wallets use 0. Config default = 1 (web-created accounts).
                sig_type = config.get("signature_type", 1)

                if (self._api_secret and not self._api_secret.startswith("${")
                        and self._api_passphrase and not self._api_passphrase.startswith("${")):
                    # Full L2 auth — can post and cancel orders
                    creds = ApiCreds(
                        api_key=self._api_key,
                        api_secret=self._api_secret,
                        api_passphrase=self._api_passphrase,
                    )
                    self._clob = ClobClient(
                        CLOB_HOST, key=self._private_key,
                        chain_id=self._chain_id, creds=creds,
                        signature_type=sig_type,
                    )
                    self._live_enabled = True
                    logger.info("Polymarket CLOB client initialized (L2 auth — live orders enabled, sig_type=%d)", sig_type)
                else:
                    # L1 only — can derive creds but not trade yet
                    self._clob = ClobClient(
                        CLOB_HOST, key=self._private_key,
                        chain_id=self._chain_id,
                        signature_type=sig_type,
                    )
                    logger.info("Polymarket CLOB client initialized (L1 auth only — derive creds with derive_api_creds())")
            except ImportError:
                logger.warning("py-clob-client not installed — live orders disabled")
            except Exception:
                logger.exception("Failed to initialize CLOB client")
        else:
            logger.info("Polymarket running in paper mode (no private key configured)")

        logger.info("Polymarket client connected")

    async def disconnect(self):
        self._running = False
        self._user_ws_connected = False
        for ws in (self._ws_market, self._ws_user):
            if ws and not ws.closed:
                await ws.close()
        for task in (self._ws_task, self._user_ws_task):
            if task and not task.done():
                task.cancel()
        if self._session:
            await self._session.close()
        logger.info("Polymarket client disconnected")

    # --- Callbacks ---

    def on_book_update(self, callback: Callable):
        self._on_book_update = callback

    def on_trade(self, callback: Callable):
        self._on_trade = callback

    def on_user_trade(self, callback: Callable):
        """Set callback for user WS fill/order events (separate from market trades)."""
        self._on_user_trade = callback

    def set_notify(self, callback: Callable):
        """Set async notification callback for WS status changes."""
        self._notify_callback = callback

    @property
    def user_ws_connected(self) -> bool:
        """True when user WS is connected and receiving fills."""
        return self._user_ws_connected

    # --- Market discovery (REST, call sparingly: 60 req/min) ---

    async def fetch_active_markets(self, limit: int = 100) -> list[Market]:
        """Fetch active markets from Gamma API."""
        markets: list[Market] = []
        offset = 0
        while True:
            params = {"limit": limit, "offset": offset, "closed": "false", "active": "true"}
            data = await self._get(f"{GAMMA_HOST}/markets", params)
            if not data:
                break
            for m in data:
                tokens = []
                outcomes = m.get("outcomes", [])
                prices = m.get("outcomePrices", [])
                clob_ids = m.get("clobTokenIds", [])
                # Gamma API returns these as JSON strings, not lists
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)
                if isinstance(prices, str):
                    prices = json.loads(prices)
                if isinstance(clob_ids, str):
                    clob_ids = json.loads(clob_ids)
                for i, outcome in enumerate(outcomes):
                    if i < len(clob_ids):
                        tokens.append({
                            "token_id": clob_ids[i],
                            "outcome": outcome,
                            "price": float(prices[i]) if i < len(prices) else 0.0,
                        })
                market = Market(
                    id=m.get("id", ""),
                    question=m.get("question", ""),
                    slug=m.get("slug", ""),
                    active=m.get("active", False),
                    end_date=m.get("endDate", ""),
                    tokens=tokens,
                    volume=float(m.get("volume", 0)),
                    liquidity=float(m.get("liquidity", 0)),
                    category=m.get("category", ""),
                    fee=float(m.get("fee", 0)),
                )
                markets.append(market)
                self._markets[market.id] = market
            if len(data) < limit:
                break
            offset += limit
        logger.info("Fetched %d active markets", len(markets))
        return markets

    async def fetch_order_book(self, token_id: str) -> OrderBook:
        """Fetch order book snapshot via REST."""
        data = await self._get(f"{CLOB_HOST}/book", {"token_id": token_id})
        if not data:
            return OrderBook(market_id="", token_id=token_id)
        return self._parse_book(token_id, data)

    # --- WebSocket: Market channel (public, no auth) ---

    async def subscribe_market(self, token_ids: list[str]):
        """Subscribe to market data via WebSocket."""
        new_tokens = [t for t in token_ids if t not in self._subscribed_tokens]
        if not new_tokens:
            return
        self._subscribed_tokens.update(new_tokens)

        if self._ws_task is None or self._ws_task.done():
            self._ws_task = asyncio.create_task(self._market_ws_loop())
        elif self._ws_market and not self._ws_market.closed:
            await self._ws_market.send_json({
                "assets_ids": new_tokens,
                "type": "market",
                "custom_feature_enabled": True,
            })
            logger.info("Subscribed to %d additional tokens", len(new_tokens))

    async def _market_ws_loop(self):
        """WebSocket loop with exponential backoff reconnect."""
        attempt = 0
        while self._running:
            connected_at = time.time()
            try:
                async with self._session.ws_connect(WS_MARKET, heartbeat=30) as ws:
                    self._ws_market = ws
                    if self._subscribed_tokens:
                        await ws.send_json({
                            "assets_ids": list(self._subscribed_tokens),
                            "type": "market",
                            "custom_feature_enabled": True,
                        })
                        logger.info("Market WS connected, subscribed to %d tokens", len(self._subscribed_tokens))

                    async for raw in ws:
                        if raw.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_market_msg(raw.data)
                        elif raw.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                            break

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Market WS error")

            if not self._running:
                break

            # Notify on disconnect
            if self._notify_callback:
                try:
                    await self._notify_callback("⚠️ Market WS disconnected — reconnecting...")
                except Exception:
                    pass

            # Reset backoff if connection was stable for > 30s
            if time.time() - connected_at > 30:
                attempt = 0

            delay = min(2 ** attempt, 60) * random.uniform(0.8, 1.2)
            logger.info("Market WS reconnecting in %.1fs (attempt %d)...", delay, attempt + 1)
            await asyncio.sleep(delay)
            attempt += 1

    async def _handle_market_msg(self, raw_data: str):
        try:
            msgs = json.loads(raw_data)
            if not isinstance(msgs, list):
                msgs = [msgs]
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from market WS")
            return

        for data in msgs:
            event_type = data.get("event_type", "")
            token_id = data.get("asset_id", "")

            if event_type == "book":
                book = self._parse_book(token_id, data)
                self._order_books[token_id] = book
                if self._on_book_update:
                    await self._on_book_update(token_id, book)

            elif event_type == "price_change":
                if token_id in self._order_books:
                    book = self._order_books[token_id]
                    self._apply_price_changes(book, data.get("changes", []))
                    book.timestamp = time.time()
                    if self._on_book_update:
                        await self._on_book_update(token_id, book)

            elif event_type == "last_trade_price":
                if self._on_trade:
                    await self._on_trade(data)

    # --- WebSocket: User channel (authenticated) ---

    async def subscribe_user(self):
        """Subscribe to user events (order fills, placements)."""
        if not self._api_key:
            logger.warning("No API credentials, skipping user WS")
            return
        if self._user_ws_task is None or self._user_ws_task.done():
            self._user_ws_task = asyncio.create_task(self._user_ws_loop())

    async def _user_ws_loop(self):
        """User WS loop with exponential backoff reconnect."""
        attempt = 0
        while self._running:
            connected_at = time.time()
            try:
                async with self._session.ws_connect(WS_USER, heartbeat=30) as ws:
                    self._ws_user = ws
                    await ws.send_json({
                        "auth": {
                            "apiKey": self._api_key,
                            "secret": self._api_secret,
                            "passphrase": self._api_passphrase,
                        },
                        "type": "user",
                    })
                    self._user_ws_connected = True
                    logger.info("User WS connected")

                    async for raw in ws:
                        if raw.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_user_msg(raw.data)
                        elif raw.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            break

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("User WS error")

            self._user_ws_connected = False

            if not self._running:
                break

            # Notify on disconnect
            if self._notify_callback:
                try:
                    await self._notify_callback("⚠️ User WS disconnected — fills may be missed!")
                except Exception:
                    pass

            # Reset backoff if connection was stable for > 30s
            if time.time() - connected_at > 30:
                attempt = 0

            delay = min(2 ** attempt, 60) * random.uniform(0.8, 1.2)
            logger.info("User WS reconnecting in %.1fs (attempt %d)...", delay, attempt + 1)
            await asyncio.sleep(delay)
            attempt += 1

    async def _handle_user_msg(self, raw_data: str):
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            return

        event_type = data.get("event_type", "")
        if event_type in ("trade", "order"):
            logger.info("User WS event: %s (status=%s, asset=%s)",
                        event_type, data.get("status", "?"), data.get("asset_id", "?")[:12])
            if self._on_user_trade:
                await self._on_user_trade(data)
        elif event_type:
            logger.debug("User WS event ignored: %s", event_type)

    # --- Order placement (REST) ---

    async def place_order(
        self, token_id: str, side: str, price: float, size: float, order_type: str = "GTC",
    ) -> dict:
        """Place a limit order. Uses py-clob-client for live, simulates for paper.

        Args:
            token_id: Conditional token ID
            side: "BUY" or "SELL"
            price: Price between 0 and 1
            size: Number of shares
            order_type: "GTC", "GTD", "FOK"
        """
        logger.info("Order: %s %.1f shares @ %.4f (token: %.8s...)", side.upper(), size, price, token_id)

        if self._live_enabled and self._clob:
            # Rate limit check — drop orders that would exceed the limit
            now = time.time()
            self._order_timestamps = [t for t in self._order_timestamps if now - t < 60]
            if len(self._order_timestamps) >= self._order_rate_limit:
                logger.warning("Rate limit: %d orders in last 60s, dropping order", len(self._order_timestamps))
                return {}
            self._order_timestamps.append(now)
            return await self._place_live_order(token_id, side, price, size, order_type)

        # Paper mode — simulate instant fill
        return {
            "order_id": f"paper_{int(time.time())}",
            "status": "paper",
            "token_id": token_id,
            "side": side.upper(),
            "price": price,
            "size": size,
        }

    async def _place_live_order(
        self, token_id: str, side: str, price: float, size: float, order_type: str,
    ) -> dict:
        """Place a signed order via py-clob-client (runs in executor to avoid blocking)."""
        from py_clob_client.clob_types import OrderArgs, OrderType as ClobOrderType
        from py_clob_client.order_builder.constants import BUY, SELL

        loop = asyncio.get_running_loop()

        clob_side = BUY if side.upper() == "BUY" else SELL
        order_args = OrderArgs(
            price=price,
            size=size,
            side=clob_side,
            token_id=token_id,
        )

        try:
            # create_order and post_order are synchronous — run in thread
            signed = await loop.run_in_executor(None, self._clob.create_order, order_args)

            otype_map = {"GTC": ClobOrderType.GTC, "GTD": ClobOrderType.GTD, "FOK": ClobOrderType.FOK}
            clob_type = otype_map.get(order_type, ClobOrderType.GTC)

            resp = await loop.run_in_executor(None, self._clob.post_order, signed, clob_type)

            order_id = resp.get("orderID", resp.get("order_id", ""))
            logger.info("Live order placed: %s (id: %s)", resp.get("status", "unknown"), order_id)

            return {
                "order_id": order_id,
                "status": resp.get("status", "live"),
                "token_id": token_id,
                "side": side.upper(),
                "price": price,
                "size": size,
                "raw_response": resp,
            }
        except Exception:
            logger.exception("Failed to place live order")
            return {}

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order by ID."""
        logger.info("Cancel order: %s", order_id)
        if order_id.startswith("paper_"):
            return True

        if self._live_enabled and self._clob:
            loop = asyncio.get_running_loop()
            try:
                await loop.run_in_executor(None, self._clob.cancel, order_id)
                logger.info("Order cancelled: %s", order_id)
                return True
            except Exception:
                logger.exception("Failed to cancel order %s", order_id)
                return False
        return True

    async def cancel_all_orders(self) -> int:
        """Cancel all open orders."""
        logger.info("Cancel all orders")
        if self._live_enabled and self._clob:
            loop = asyncio.get_running_loop()
            try:
                resp = await loop.run_in_executor(None, self._clob.cancel_all)
                cancelled = len(resp) if isinstance(resp, list) else 0
                logger.info("Cancelled %d orders", cancelled)
                return cancelled
            except Exception:
                logger.exception("Failed to cancel all orders")
                return 0
        return 0

    async def derive_api_creds(self) -> dict:
        """Derive API credentials from private key (one-time setup helper)."""
        if not self._clob:
            logger.error("CLOB client not initialized — need private key")
            return {}
        loop = asyncio.get_running_loop()
        try:
            creds = await loop.run_in_executor(None, self._clob.create_or_derive_api_creds)
            logger.info("API credentials derived successfully")
            return {
                "api_key": creds.api_key,
                "api_secret": creds.api_secret,
                "api_passphrase": creds.api_passphrase,
            }
        except Exception:
            logger.exception("Failed to derive API credentials")
            return {}

    # --- Queries ---

    def get_order_book(self, token_id: str) -> Optional[OrderBook]:
        return self._order_books.get(token_id)

    def get_market(self, market_id: str) -> Optional[Market]:
        return self._markets.get(market_id)

    # --- Internal ---

    def _parse_book(self, token_id: str, data: dict) -> OrderBook:
        bids = sorted(
            [OrderBookLevel(price=float(b["price"]), size=float(b["size"])) for b in data.get("bids", [])],
            key=lambda x: x.price, reverse=True,
        )
        asks = sorted(
            [OrderBookLevel(price=float(a["price"]), size=float(a["size"])) for a in data.get("asks", [])],
            key=lambda x: x.price,
        )
        return OrderBook(
            market_id=data.get("market", ""), token_id=token_id,
            bids=bids, asks=asks, timestamp=time.time(),
        )

    def _apply_price_changes(self, book: OrderBook, changes: list):
        for change in changes:
            side = change.get("side", "").upper()
            price = float(change.get("price", 0))
            size = float(change.get("size", 0))

            levels = book.bids if side == "BUY" else book.asks
            levels[:] = [lvl for lvl in levels if abs(lvl.price - price) > 1e-9]
            if size > 0:
                levels.append(OrderBookLevel(price=price, size=size))

            if side == "BUY":
                levels.sort(key=lambda x: x.price, reverse=True)
            else:
                levels.sort(key=lambda x: x.price)

    async def _get(self, url: str, params: dict = None) -> Optional[dict | list]:
        try:
            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                if resp.status == 429:
                    logger.warning("Rate limited on %s", url)
                    await asyncio.sleep(5)
                    return None
                logger.error("GET %s → %d", url, resp.status)
                return None
        except asyncio.TimeoutError:
            logger.error("Timeout: GET %s", url)
            return None
        except Exception:
            logger.exception("Error: GET %s", url)
            return None
