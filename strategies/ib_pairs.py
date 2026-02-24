"""IB Pairs Trading — Dynamic cointegration scanner + z-score mean reversion.

Scans a configurable universe for cointegrated pairs using Engle-Granger.
Filters by half-life (5-60 days) and ranks by statistical strength.
Trades the top N pairs with z-score entry/exit/stop thresholds.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from connectors.ib_client import IBClient
from core.portfolio import Platform, Portfolio, Side
from core.risk_manager import RiskManager
from strategies.base_strategy import BaseStrategy, Signal

logger = logging.getLogger(__name__)


@dataclass
class PairStats:
    """Cointegration statistics for a pair."""
    symbol_a: str
    symbol_b: str
    pvalue: float           # Engle-Granger p-value (lower = better)
    hedge_ratio: float      # OLS beta: B = hedge_ratio * A + residual
    half_life: float        # mean reversion half-life in days
    residual_std: float     # std of spread residuals
    spread_mean: float      # mean of spread


@dataclass
class ActivePair:
    """State for a pair currently being traded."""
    stats: PairStats
    zscore: float = 0.0
    position_side: str = ""  # "" = flat, "long" = long spread, "short" = short spread
    entry_zscore: float = 0.0
    entry_time: float = 0.0
    prices_a: list[float] = field(default_factory=list)
    prices_b: list[float] = field(default_factory=list)


class IBPairsTrader(BaseStrategy):
    """Pairs trading with dynamic cointegration scanning.

    Lifecycle:
        1. on_start(): Load historical data, run cointegration scan, subscribe to quotes
        2. evaluate(): Compute z-scores, generate entry/exit signals
        3. Rescan every N hours to rotate stale pairs
    """

    def __init__(self, name: str, portfolio: Portfolio, risk_manager: RiskManager,
                 config: dict, ib_client: IBClient):
        super().__init__(name, portfolio, risk_manager, config)
        self.ib = ib_client

        # Config
        ib_cfg = config
        scanner_cfg = ib_cfg.get("scanner", {})

        self._universe = self._build_universe(ib_cfg.get("universe", {}))
        self._lookback_days = scanner_cfg.get("lookback_days", 120)
        self._rescan_hours = scanner_cfg.get("rescan_hours", 24)
        self._max_active = scanner_cfg.get("max_active_pairs", 5)
        self._min_half_life = scanner_cfg.get("min_half_life", 5)
        self._max_half_life = scanner_cfg.get("max_half_life", 60)
        self._zscore_entry = ib_cfg.get("zscore_entry", 2.0)
        self._zscore_exit = ib_cfg.get("zscore_exit", 0.5)
        self._zscore_stop = ib_cfg.get("zscore_stop", 3.5)
        self._coint_pvalue = ib_cfg.get("cointegration_pvalue", 0.05)

        # State
        self._active_pairs: dict[str, ActivePair] = {}  # "A:B" -> ActivePair
        self._historical: dict[str, np.ndarray] = {}     # symbol -> close prices
        self._last_scan_time: float = 0
        self._position_size_usd = 50.0  # per leg, conservative for paper

    @staticmethod
    def _build_universe(universe_cfg: dict) -> list[str]:
        """Flatten universe config into unique symbol list."""
        symbols = set()
        for category, items in universe_cfg.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, list):
                        symbols.update(item)
                    else:
                        symbols.add(item)
        return sorted(symbols)

    # --- Lifecycle ---

    async def on_start(self):
        if not self.ib.is_connected:
            self.logger.warning("IB not connected — pairs trading disabled")
            self.enabled = False
            return

        self.logger.info("Loading historical data for %d symbols...", len(self._universe))
        await self._load_historical()

        self.logger.info("Running initial cointegration scan...")
        await self._scan_and_activate()

        # Subscribe to real-time quotes for active pair symbols
        active_symbols = self._get_active_symbols()
        if active_symbols:
            await self.ib.subscribe(active_symbols)
            self.logger.info("Subscribed to %d symbols for pairs trading", len(active_symbols))

    async def on_stop(self):
        self.logger.info("Pairs trader stopping, %d active pairs", len(self._active_pairs))

    # --- Evaluation ---

    async def evaluate(self) -> list[Signal]:
        if not self.ib.is_connected or not self._active_pairs:
            return []

        # Periodic rescan
        if time.time() - self._last_scan_time > self._rescan_hours * 3600:
            self.logger.info("Periodic cointegration rescan...")
            await self._load_historical()
            await self._scan_and_activate()
            # Resubscribe
            active_symbols = self._get_active_symbols()
            if active_symbols:
                await self.ib.subscribe(active_symbols)

        signals = []
        for key, pair in list(self._active_pairs.items()):
            price_a = self.ib.get_mid_price(pair.stats.symbol_a)
            price_b = self.ib.get_mid_price(pair.stats.symbol_b)

            if price_a <= 0 or price_b <= 0:
                continue

            # Calculate spread and z-score
            spread = price_b - pair.stats.hedge_ratio * price_a
            pair.zscore = (spread - pair.stats.spread_mean) / max(pair.stats.residual_std, 0.001)

            sigs = self._evaluate_pair(pair)
            signals.extend(sigs)

        return signals

    def _evaluate_pair(self, pair: ActivePair) -> list[Signal]:
        """Generate signals for a single pair based on z-score."""
        z = pair.zscore
        signals = []

        if pair.position_side == "":
            # Entry: z-score crossed threshold
            if z >= self._zscore_entry:
                # Spread is too high → short spread (sell B, buy A)
                signals.extend(self._create_pair_signals(pair, "short_spread"))
            elif z <= -self._zscore_entry:
                # Spread is too low → long spread (buy B, sell A)
                signals.extend(self._create_pair_signals(pair, "long_spread"))

        elif pair.position_side == "short_spread":
            # Exit: z-score reverted
            if z <= self._zscore_exit:
                signals.extend(self._create_close_signals(pair))
            # Stop: z-score blew out
            elif z >= self._zscore_stop:
                self.logger.warning(
                    "Stop hit for %s:%s (z=%.2f)",
                    pair.stats.symbol_a, pair.stats.symbol_b, z,
                )
                signals.extend(self._create_close_signals(pair))

        elif pair.position_side == "long_spread":
            if z >= -self._zscore_exit:
                signals.extend(self._create_close_signals(pair))
            elif z <= -self._zscore_stop:
                self.logger.warning(
                    "Stop hit for %s:%s (z=%.2f)",
                    pair.stats.symbol_a, pair.stats.symbol_b, z,
                )
                signals.extend(self._create_close_signals(pair))

        return signals

    def _create_pair_signals(self, pair: ActivePair, direction: str) -> list[Signal]:
        """Create entry signals for a pair trade."""
        price_a = self.ib.get_mid_price(pair.stats.symbol_a)
        price_b = self.ib.get_mid_price(pair.stats.symbol_b)
        if price_a <= 0 or price_b <= 0:
            return []

        # Size: equal dollar amount per leg
        size = self._position_size_usd

        if direction == "short_spread":
            # Short B (overpriced), Long A (underpriced)
            pair.position_side = "short_spread"
            pair.entry_zscore = pair.zscore
            pair.entry_time = time.time()
            return [
                Signal(
                    strategy=self.name,
                    market_id=pair.stats.symbol_a,
                    symbol=pair.stats.symbol_a,
                    direction="buy",
                    size_usd=size,
                    price=price_a,
                    confidence=min(abs(pair.zscore) / self._zscore_stop, 1.0),
                    metadata={
                        "platform": "ib",
                        "pair": f"{pair.stats.symbol_a}:{pair.stats.symbol_b}",
                        "leg": "A",
                        "zscore": pair.zscore,
                    },
                ),
                Signal(
                    strategy=self.name,
                    market_id=pair.stats.symbol_b,
                    symbol=pair.stats.symbol_b,
                    direction="sell",
                    size_usd=size,
                    price=price_b,
                    confidence=min(abs(pair.zscore) / self._zscore_stop, 1.0),
                    metadata={
                        "platform": "ib",
                        "pair": f"{pair.stats.symbol_a}:{pair.stats.symbol_b}",
                        "leg": "B",
                        "zscore": pair.zscore,
                    },
                ),
            ]
        else:
            # Long B (underpriced), Short A (overpriced)
            pair.position_side = "long_spread"
            pair.entry_zscore = pair.zscore
            pair.entry_time = time.time()
            return [
                Signal(
                    strategy=self.name,
                    market_id=pair.stats.symbol_a,
                    symbol=pair.stats.symbol_a,
                    direction="sell",
                    size_usd=size,
                    price=price_a,
                    confidence=min(abs(pair.zscore) / self._zscore_stop, 1.0),
                    metadata={
                        "platform": "ib",
                        "pair": f"{pair.stats.symbol_a}:{pair.stats.symbol_b}",
                        "leg": "A",
                        "zscore": pair.zscore,
                    },
                ),
                Signal(
                    strategy=self.name,
                    market_id=pair.stats.symbol_b,
                    symbol=pair.stats.symbol_b,
                    direction="buy",
                    size_usd=size,
                    price=price_b,
                    confidence=min(abs(pair.zscore) / self._zscore_stop, 1.0),
                    metadata={
                        "platform": "ib",
                        "pair": f"{pair.stats.symbol_a}:{pair.stats.symbol_b}",
                        "leg": "B",
                        "zscore": pair.zscore,
                    },
                ),
            ]

    def _create_close_signals(self, pair: ActivePair) -> list[Signal]:
        """Create exit signals to flatten a pair position."""
        price_a = self.ib.get_mid_price(pair.stats.symbol_a)
        price_b = self.ib.get_mid_price(pair.stats.symbol_b)
        if price_a <= 0 or price_b <= 0:
            return []

        size = self._position_size_usd

        if pair.position_side == "short_spread":
            # Was: long A, short B → close: sell A, buy B
            signals = [
                Signal(
                    strategy=self.name, market_id=pair.stats.symbol_a,
                    symbol=pair.stats.symbol_a, direction="sell",
                    size_usd=size, price=price_a,
                    metadata={"platform": "ib", "close": True,
                              "pair": f"{pair.stats.symbol_a}:{pair.stats.symbol_b}"},
                ),
                Signal(
                    strategy=self.name, market_id=pair.stats.symbol_b,
                    symbol=pair.stats.symbol_b, direction="buy",
                    size_usd=size, price=price_b,
                    metadata={"platform": "ib", "close": True,
                              "pair": f"{pair.stats.symbol_a}:{pair.stats.symbol_b}"},
                ),
            ]
        else:
            # Was: short A, long B → close: buy A, sell B
            signals = [
                Signal(
                    strategy=self.name, market_id=pair.stats.symbol_a,
                    symbol=pair.stats.symbol_a, direction="buy",
                    size_usd=size, price=price_a,
                    metadata={"platform": "ib", "close": True,
                              "pair": f"{pair.stats.symbol_a}:{pair.stats.symbol_b}"},
                ),
                Signal(
                    strategy=self.name, market_id=pair.stats.symbol_b,
                    symbol=pair.stats.symbol_b, direction="sell",
                    size_usd=size, price=price_b,
                    metadata={"platform": "ib", "close": True,
                              "pair": f"{pair.stats.symbol_a}:{pair.stats.symbol_b}"},
                ),
            ]

        pair.position_side = ""
        pair.entry_zscore = 0
        return signals

    # --- Cointegration Scanner ---

    async def _load_historical(self):
        """Fetch historical daily closes for entire universe."""
        tasks = []
        for symbol in self._universe:
            tasks.append(self._fetch_one(symbol))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        loaded = 0
        for symbol, result in zip(self._universe, results):
            if isinstance(result, Exception):
                self.logger.warning("Failed to load %s: %s", symbol, result)
                continue
            if result is not None and len(result) >= 30:
                self._historical[symbol] = result
                loaded += 1

        self.logger.info("Loaded historical data for %d/%d symbols", loaded, len(self._universe))

    async def _fetch_one(self, symbol: str) -> Optional[np.ndarray]:
        bars = await self.ib.get_historical(symbol, days=self._lookback_days)
        if not bars:
            return None
        closes = np.array([b["close"] for b in bars], dtype=np.float64)
        return closes

    async def _scan_and_activate(self):
        """Scan all pairs for cointegration and activate the best ones."""
        symbols_with_data = [s for s in self._universe if s in self._historical]
        if len(symbols_with_data) < 2:
            self.logger.warning("Not enough symbols with data to scan pairs")
            return

        # Test all combinations
        candidates = []
        for sym_a, sym_b in itertools.combinations(symbols_with_data, 2):
            prices_a = self._historical[sym_a]
            prices_b = self._historical[sym_b]

            # Align lengths
            min_len = min(len(prices_a), len(prices_b))
            if min_len < 30:
                continue
            pa = prices_a[-min_len:]
            pb = prices_b[-min_len:]

            stats = self._test_cointegration(sym_a, sym_b, pa, pb)
            if stats and stats.pvalue < self._coint_pvalue:
                if self._min_half_life <= stats.half_life <= self._max_half_life:
                    candidates.append(stats)

        # Rank by p-value (best first)
        candidates.sort(key=lambda s: s.pvalue)

        # Keep existing positions, fill remaining slots
        current_keys = set(self._active_pairs.keys())
        positioned_keys = {k for k, p in self._active_pairs.items() if p.position_side != ""}
        new_active = {}

        # Keep pairs with open positions
        for key in positioned_keys:
            if key in self._active_pairs:
                new_active[key] = self._active_pairs[key]

        # Add best new candidates
        for stats in candidates:
            key = f"{stats.symbol_a}:{stats.symbol_b}"
            if len(new_active) >= self._max_active:
                break
            if key not in new_active:
                new_active[key] = ActivePair(stats=stats)

        self._active_pairs = new_active
        self._last_scan_time = time.time()

        self.logger.info(
            "Scan complete: %d candidates, %d active pairs",
            len(candidates), len(self._active_pairs),
        )
        for key, pair in self._active_pairs.items():
            self.logger.info(
                "  %s: p=%.4f, hedge=%.3f, half_life=%.1fd",
                key, pair.stats.pvalue, pair.stats.hedge_ratio, pair.stats.half_life,
            )

    @staticmethod
    def _test_cointegration(
        sym_a: str, sym_b: str, prices_a: np.ndarray, prices_b: np.ndarray,
    ) -> Optional[PairStats]:
        """Run Engle-Granger cointegration test on two price series."""
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            logger.error("statsmodels not installed — cannot run cointegration test")
            return None

        # OLS regression: B = beta * A + residual
        # Using numpy for speed (avoid full statsmodels OLS overhead)
        x = np.column_stack([prices_a, np.ones(len(prices_a))])
        beta, intercept = np.linalg.lstsq(x, prices_b, rcond=None)[0]

        residuals = prices_b - beta * prices_a - intercept

        # ADF test on residuals
        try:
            adf_result = adfuller(residuals, maxlag=int(len(residuals) ** (1/3)))
        except Exception:
            return None

        pvalue = adf_result[1]

        # Half-life of mean reversion (AR(1) model on residuals)
        residuals_lag = residuals[:-1]
        residuals_now = residuals[1:]
        if len(residuals_lag) < 10:
            return None

        x_ar = np.column_stack([residuals_lag, np.ones(len(residuals_lag))])
        try:
            phi = np.linalg.lstsq(x_ar, residuals_now, rcond=None)[0][0]
        except np.linalg.LinAlgError:
            return None

        if phi >= 1.0 or phi <= 0:
            return None  # no mean reversion

        half_life = -np.log(2) / np.log(phi)

        return PairStats(
            symbol_a=sym_a,
            symbol_b=sym_b,
            pvalue=pvalue,
            hedge_ratio=beta,
            half_life=half_life,
            residual_std=float(np.std(residuals)),
            spread_mean=float(np.mean(residuals)),
        )

    def _get_active_symbols(self) -> list[str]:
        """Get unique symbols from all active pairs."""
        symbols = set()
        for pair in self._active_pairs.values():
            symbols.add(pair.stats.symbol_a)
            symbols.add(pair.stats.symbol_b)
        return sorted(symbols)
