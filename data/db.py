"""SQLite database — WAL mode, async writes, indexed by timestamp/strategy."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)

DB_PATH = "data/trading_bot.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    platform TEXT NOT NULL,
    market_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    size REAL NOT NULL,
    fee REAL DEFAULT 0,
    slippage REAL DEFAULT 0,
    strategy TEXT NOT NULL,
    pnl REAL DEFAULT 0,
    latency_ms REAL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);

CREATE TABLE IF NOT EXISTS audits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    type TEXT NOT NULL,
    strategy TEXT,
    metrics TEXT,
    notes TEXT,
    actions TEXT
);

CREATE INDEX IF NOT EXISTS idx_audits_date ON audits(date);

CREATE TABLE IF NOT EXISTS bot_state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


class Database:
    """Async SQLite database with WAL mode and write queue."""

    def __init__(self, db_path: str = DB_PATH):
        self._path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._write_queue: asyncio.Queue = asyncio.Queue()
        self._writer_task: Optional[asyncio.Task] = None
        self._running = False

    async def connect(self):
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.executescript(SCHEMA)
        await self._db.commit()

        self._running = True
        self._writer_task = asyncio.create_task(self._write_loop())
        logger.info("Database connected: %s (WAL mode)", self._path)

    async def disconnect(self):
        self._running = False
        if self._writer_task:
            # Drain remaining writes
            await self._write_queue.put(None)
            await self._writer_task
        if self._db:
            await self._db.close()
        logger.info("Database disconnected")

    # --- Trade logging (async queue) ---

    async def log_trade(self, trade, pnl: float = 0.0):
        """Queue a trade for async writing."""
        await self._write_queue.put(("trade", trade, pnl))

    async def _write_loop(self):
        """Process write queue sequentially."""
        while self._running or not self._write_queue.empty():
            try:
                item = await asyncio.wait_for(self._write_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            if item is None:
                break

            try:
                kind = item[0]
                if kind == "trade":
                    await self._insert_trade(item[1], item[2])
                elif kind == "audit":
                    await self._insert_audit(*item[1:])
            except Exception:
                logger.exception("Error writing to database")

    async def _insert_trade(self, trade, pnl: float):
        await self._db.execute(
            """INSERT INTO trades
               (trade_id, timestamp, platform, market_id, symbol, side, price, size,
                fee, slippage, strategy, pnl, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade.trade_id, trade.timestamp, trade.platform.value,
                trade.market_id, trade.symbol, trade.side.value,
                trade.price, trade.size, trade.fee, trade.slippage,
                trade.strategy, pnl, trade.latency_ms,
            ),
        )
        await self._db.commit()

    # --- Audit logging ---

    async def log_audit(self, audit_type: str, strategy: str, metrics: dict,
                        notes: str = "", actions: str = ""):
        await self._write_queue.put((
            "audit", audit_type, strategy, metrics, notes, actions,
        ))

    async def _insert_audit(self, audit_type, strategy, metrics, notes, actions):
        from datetime import datetime, timezone
        await self._db.execute(
            """INSERT INTO audits (date, type, strategy, metrics, notes, actions)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                audit_type, strategy,
                json.dumps(metrics), notes, actions,
            ),
        )
        await self._db.commit()

    # --- Bot state ---

    async def save_state(self, key: str, value: dict):
        await self._db.execute(
            """INSERT OR REPLACE INTO bot_state (key, value, updated_at)
               VALUES (?, ?, ?)""",
            (key, json.dumps(value), time.time()),
        )
        await self._db.commit()

    async def load_state(self, key: str) -> Optional[dict]:
        async with self._db.execute(
            "SELECT value FROM bot_state WHERE key = ?", (key,)
        ) as cursor:
            row = await cursor.fetchone()
            return json.loads(row[0]) if row else None

    # --- Queries ---

    async def get_trades(self, strategy: str = None, limit: int = 50) -> list[dict]:
        query = "SELECT * FROM trades"
        params = []
        if strategy:
            query += " WHERE strategy = ?"
            params.append(strategy)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        async with self._db.execute(query, params) as cursor:
            columns = [d[0] for d in cursor.description]
            rows = await cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]

    async def get_pnl_summary(self, strategy: str = None) -> dict:
        query = "SELECT SUM(pnl) as total_pnl, SUM(fee) as total_fees, COUNT(*) as trade_count FROM trades"
        params = []
        if strategy:
            query += " WHERE strategy = ?"
            params.append(strategy)

        async with self._db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return {
                "total_pnl": row[0] or 0.0,
                "total_fees": row[1] or 0.0,
                "trade_count": row[2] or 0,
            }
