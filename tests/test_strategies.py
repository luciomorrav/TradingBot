"""Tests for strategy logic — signals, cointegration, market maker inventory."""
import math
import time

import numpy as np
import pytest

from strategies.base_strategy import Signal


# --- Signal validation ---

def test_signal_valid():
    s = Signal(strategy="test", market_id="m1", symbol="SYM", direction="buy", size_usd=50, price=0.5)
    assert s.direction == "buy"
    assert s.confidence == 1.0


def test_signal_invalid_direction():
    with pytest.raises(ValueError, match="Invalid direction"):
        Signal(strategy="test", market_id="m1", symbol="SYM", direction="hold", size_usd=50, price=0.5)


def test_signal_invalid_price():
    with pytest.raises(ValueError, match="Invalid price"):
        Signal(strategy="test", market_id="m1", symbol="SYM", direction="buy", size_usd=50, price=-1)

    with pytest.raises(ValueError):
        Signal(strategy="test", market_id="m1", symbol="SYM", direction="buy", size_usd=50, price=float("inf"))


def test_signal_invalid_size():
    with pytest.raises(ValueError, match="Invalid size"):
        Signal(strategy="test", market_id="m1", symbol="SYM", direction="buy", size_usd=0, price=0.5)

    with pytest.raises(ValueError):
        Signal(strategy="test", market_id="m1", symbol="SYM", direction="buy", size_usd=-10, price=0.5)


def test_signal_invalid_confidence():
    with pytest.raises(ValueError, match="Confidence"):
        Signal(strategy="test", market_id="m1", symbol="SYM", direction="buy", size_usd=50, price=0.5, confidence=1.5)


def test_signal_metadata_default():
    s = Signal(strategy="test", market_id="m1", symbol="SYM", direction="buy", size_usd=50, price=0.5)
    assert s.metadata == {}


# --- Cointegration tests ---

def test_cointegration_on_cointegrated_pair():
    """Two series with known cointegration should have low p-value."""
    from strategies.ib_pairs import IBPairsTrader

    np.random.seed(42)
    n = 300
    # Create cointegrated pair: A and B share a common trend
    # with a mean-reverting spread (OU process with phi=0.95)
    a = np.cumsum(np.random.randn(n) * 0.5) + 100
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.95 * spread[i - 1] + np.random.randn() * 0.5
    b = 1.5 * a + spread + 50

    stats = IBPairsTrader._test_cointegration("A", "B", a, b)
    assert stats is not None, "Cointegration test returned None for known cointegrated pair"
    assert stats.pvalue < 0.10  # should be cointegrated (may not always hit 0.05 with noise)
    assert 1.2 < stats.hedge_ratio < 1.8  # close to 1.5
    assert stats.half_life > 0


def test_cointegration_on_random_pair():
    """Two random walks should NOT be cointegrated."""
    from strategies.ib_pairs import IBPairsTrader

    np.random.seed(123)
    n = 200
    a = np.cumsum(np.random.randn(n)) + 100
    b = np.cumsum(np.random.randn(n)) + 100  # independent random walk

    stats = IBPairsTrader._test_cointegration("A", "B", a, b)
    # Should either be None or have high p-value
    if stats is not None:
        assert stats.pvalue > 0.05


def test_cointegration_half_life_reasonable():
    """Half-life should be in reasonable range for a mean-reverting spread."""
    from strategies.ib_pairs import IBPairsTrader

    np.random.seed(7)
    n = 300
    a = np.cumsum(np.random.randn(n) * 0.5) + 100
    # B reverts to 2*A with half-life ~10 bars
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = 0.9 * spread[i - 1] + np.random.randn() * 0.3
    b = 2 * a + spread + 50

    stats = IBPairsTrader._test_cointegration("A", "B", a, b)
    assert stats is not None
    assert stats.half_life > 1  # at least 1 day
    assert stats.half_life < 100  # not absurdly long


# --- Market maker Avellaneda-Stoikov ---

def test_avellaneda_stoikov_basic():
    """A-S model should produce bid < mid < ask."""
    from connectors.polymarket_client import OrderBook, OrderBookLevel
    from strategies.poly_market_maker import MarketState, PolyMarketMaker
    from core.portfolio import Portfolio
    from core.risk_manager import RiskConfig, RiskManager
    from unittest.mock import MagicMock

    portfolio = Portfolio(capital_usd=500)
    rm = RiskManager(portfolio, RiskConfig())
    client = MagicMock()

    mm = PolyMarketMaker("test", portfolio, rm, {"market_maker": {}}, client)

    book = OrderBook(
        market_id="m1", token_id="t1",
        bids=[OrderBookLevel(0.45, 100)],
        asks=[OrderBookLevel(0.55, 100)],
    )
    state = MarketState(token_id="t1", market_id="m1", outcome="YES")
    for _ in range(10):
        state.mid_prices.append(0.50)

    bid, ask = mm._avellaneda_stoikov(book, state)
    assert bid is not None and ask is not None
    assert bid < book.mid_price < ask
    assert 0.01 <= bid <= 0.99
    assert 0.01 <= ask <= 0.99


def test_avellaneda_stoikov_inventory_skew():
    """Long inventory should lower reservation price → bid lower, ask lower."""
    from connectors.polymarket_client import OrderBook, OrderBookLevel
    from strategies.poly_market_maker import MarketState, PolyMarketMaker
    from core.portfolio import Portfolio
    from core.risk_manager import RiskConfig, RiskManager
    from unittest.mock import MagicMock

    portfolio = Portfolio(capital_usd=500)
    rm = RiskManager(portfolio, RiskConfig())
    client = MagicMock()

    mm = PolyMarketMaker("test", portfolio, rm, {"market_maker": {}}, client)

    book = OrderBook(
        market_id="m1", token_id="t1",
        bids=[OrderBookLevel(0.45, 100)],
        asks=[OrderBookLevel(0.55, 100)],
    )

    # Flat inventory
    state_flat = MarketState(token_id="t1", market_id="m1", outcome="YES", inventory=0)
    for _ in range(10):
        state_flat.mid_prices.append(0.50)
    bid_flat, ask_flat = mm._avellaneda_stoikov(book, state_flat)

    # Moderate long inventory (should push prices down to sell more)
    state_long = MarketState(token_id="t1", market_id="m1", outcome="YES", inventory=50)
    for _ in range(10):
        state_long.mid_prices.append(0.50)
    bid_long, ask_long = mm._avellaneda_stoikov(book, state_long)

    assert bid_long is not None and ask_long is not None
    # Reservation price drops with positive inventory → both bid and ask shift down
    assert bid_long <= bid_flat
    assert ask_long <= ask_flat


def test_avellaneda_stoikov_extreme_mid_returns_none():
    """Mid price near 0 or 1 should return None (can't quote)."""
    from connectors.polymarket_client import OrderBook, OrderBookLevel
    from strategies.poly_market_maker import MarketState, PolyMarketMaker
    from core.portfolio import Portfolio
    from core.risk_manager import RiskConfig, RiskManager
    from unittest.mock import MagicMock

    portfolio = Portfolio(capital_usd=500)
    rm = RiskManager(portfolio, RiskConfig())
    client = MagicMock()

    mm = PolyMarketMaker("test", portfolio, rm, {"market_maker": {}}, client)

    book = OrderBook(
        market_id="m1", token_id="t1",
        bids=[OrderBookLevel(0.01, 100)],
        asks=[OrderBookLevel(0.02, 100)],
    )
    state = MarketState(token_id="t1", market_id="m1", outcome="YES")
    for _ in range(10):
        state.mid_prices.append(0.015)

    bid, ask = mm._avellaneda_stoikov(book, state)
    assert bid is None and ask is None


# --- Market maker inventory tracking ---

def test_track_order_paper_updates_inventory():
    """Paper orders should immediately update inventory."""
    from strategies.poly_market_maker import MarketState, PolyMarketMaker
    from core.portfolio import Portfolio
    from core.risk_manager import RiskConfig, RiskManager
    from unittest.mock import MagicMock

    portfolio = Portfolio(capital_usd=500)
    rm = RiskManager(portfolio, RiskConfig())
    client = MagicMock()

    mm = PolyMarketMaker("test", portfolio, rm, {"market_maker": {}}, client)
    mm.market_states["t1"] = MarketState(token_id="t1", market_id="m1", outcome="YES")

    mm.track_order("paper_123", "t1", "buy", price=0.50, size=100, size_usd=50)
    assert mm.market_states["t1"].inventory == 100.0  # 50 / 0.50

    mm.track_order("paper_124", "t1", "sell", price=0.50, size=60, size_usd=30)
    assert mm.market_states["t1"].inventory == 40.0  # 100 - 60


def test_track_order_live_creates_active_order():
    """Live orders should be tracked in _active_orders, not update inventory."""
    from strategies.poly_market_maker import MarketState, PolyMarketMaker
    from core.portfolio import Portfolio
    from core.risk_manager import RiskConfig, RiskManager
    from unittest.mock import MagicMock

    portfolio = Portfolio(capital_usd=500)
    rm = RiskManager(portfolio, RiskConfig())
    client = MagicMock()

    mm = PolyMarketMaker("test", portfolio, rm, {"market_maker": {}}, client)
    mm.market_states["t1"] = MarketState(token_id="t1", market_id="m1", outcome="YES")

    mm.track_order("0xabc123", "t1", "buy", price=0.50, size=100, size_usd=50)
    assert mm.market_states["t1"].inventory == 0.0  # not updated yet
    assert "0xabc123" in mm._active_orders


def test_on_fill_updates_inventory():
    """on_fill should update inventory from actual fills."""
    from strategies.poly_market_maker import MarketState, PolyMarketMaker
    from core.portfolio import Portfolio
    from core.risk_manager import RiskConfig, RiskManager
    from unittest.mock import MagicMock

    portfolio = Portfolio(capital_usd=500)
    rm = RiskManager(portfolio, RiskConfig())
    client = MagicMock()

    mm = PolyMarketMaker("test", portfolio, rm, {"market_maker": {}}, client)
    mm.market_states["t1"] = MarketState(token_id="t1", market_id="m1", outcome="YES")

    # Track a live order first
    mm.track_order("0xabc123", "t1", "buy", price=0.50, size=100, size_usd=50)

    # Simulate fill
    mm.on_fill("0xabc123", filled_size=100, filled_price=0.50)
    assert mm.market_states["t1"].inventory == 100.0
    assert "0xabc123" not in mm._active_orders  # removed after fill
