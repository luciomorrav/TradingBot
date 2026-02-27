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


# --- Market selection scoring ---

def _make_mm():
    """Helper: create a PolyMarketMaker for scoring tests."""
    from strategies.poly_market_maker import PolyMarketMaker
    from core.portfolio import Portfolio
    from core.risk_manager import RiskConfig, RiskManager
    from unittest.mock import MagicMock

    portfolio = Portfolio(capital_usd=500)
    rm = RiskManager(portfolio, RiskConfig())
    client = MagicMock()
    return PolyMarketMaker("test", portfolio, rm, {"market_maker": {}}, client)


def _make_market(mid=0.5, volume=2000, liquidity=500, end_date="2026-06-01T00:00:00Z", **kw):
    """Helper: create a Market with controllable params."""
    from connectors.polymarket_client import Market
    return Market(
        id=kw.get("id", "m1"),
        question=kw.get("question", "Test market?"),
        slug="test-market",
        active=kw.get("active", True),
        end_date=end_date,
        tokens=[
            {"token_id": "t1", "outcome": "Yes", "price": mid},
            {"token_id": "t2", "outcome": "No", "price": round(1 - mid, 4)},
        ],
        volume=volume,
        liquidity=liquidity,
        fee=kw.get("fee", 0.02),
    )


def test_score_market_balanced():
    """A balanced 50/50 market with good volume/liq should score high."""
    mm = _make_mm()
    m = _make_market(mid=0.50, volume=2000, liquidity=500)
    score = mm._score_market(m)
    assert score > 0.7, f"Balanced market score {score} should be > 0.7"


def test_score_market_extreme_price():
    """A 95/5 market should score much lower than a 50/50."""
    mm = _make_mm()
    m_extreme = _make_market(mid=0.95, volume=2000, liquidity=500)
    m_balanced = _make_market(mid=0.50, volume=2000, liquidity=500)
    score_extreme = mm._score_market(m_extreme)
    score_balanced = mm._score_market(m_balanced)
    assert score_extreme < score_balanced, "Extreme-price market should score lower"
    assert score_extreme < 0.7, f"Extreme-price score {score_extreme} should be < 0.7"


def test_select_markets_ordering():
    """Markets should be selected in descending score order, limited to max_markets."""
    mm = _make_mm()
    mm.max_markets = 3
    markets = [
        _make_market(mid=0.50, volume=2000, liquidity=500, id="best", question="Balanced?"),
        _make_market(mid=0.90, volume=500, liquidity=100, id="worst", question="Extreme low vol?"),
        _make_market(mid=0.45, volume=1500, liquidity=400, id="good", question="Slightly off?"),
        _make_market(mid=0.30, volume=800, liquidity=200, id="ok", question="Off-center?"),
        _make_market(mid=0.10, volume=300, liquidity=60, id="bad", question="Very skewed?"),
    ]
    selected = mm._select_markets(markets)
    assert len(selected) == 3
    ids = [m.id for m in selected]
    assert ids[0] == "best", f"Best market should be first, got {ids}"
    # Worst/bad should not be in top 3
    assert "worst" not in ids
    assert "bad" not in ids


@pytest.mark.asyncio
async def test_validate_markets_drops_tight_spread():
    """Markets with spread < min should be dropped during post-WS validation."""
    from connectors.polymarket_client import OrderBook, OrderBookLevel
    from strategies.poly_market_maker import MarketState
    from unittest.mock import MagicMock, AsyncMock

    mm = _make_mm()
    mm.client.subscribe_market = AsyncMock()

    # Set up two tokens: one with wide spread, one with tight spread
    mm.market_states["t_wide"] = MarketState(token_id="t_wide", market_id="m1", outcome="Wide")
    mm.market_states["t_tight"] = MarketState(token_id="t_tight", market_id="m2", outcome="Tight")

    def mock_book(tid):
        if tid == "t_wide":
            return OrderBook("m1", "t_wide",
                             bids=[OrderBookLevel(0.44, 100)],
                             asks=[OrderBookLevel(0.56, 100)])  # spread 0.12
        elif tid == "t_tight":
            return OrderBook("m2", "t_tight",
                             bids=[OrderBookLevel(0.495, 100)],
                             asks=[OrderBookLevel(0.500, 100)])  # spread 0.005
        return None

    mm.client.get_order_book = mock_book

    await mm._validate_markets()

    assert "t_wide" in mm.market_states, "Wide-spread market should remain"
    assert "t_tight" not in mm.market_states, "Tight-spread market should be dropped"


# --- Portfolio persistence ---

@pytest.mark.asyncio
async def test_portfolio_to_dict_and_restore():
    """Portfolio serializes and restores correctly."""
    from core.portfolio import Portfolio, Platform, Side, Trade
    import time as _time

    p = Portfolio(capital_usd=500)
    trade = Trade(
        trade_id="t1", platform=Platform.POLYMARKET, market_id="tok1",
        symbol="Yes", side=Side.BUY, price=0.50, size=21.0, fee=0.10,
        strategy="poly_mm", timestamp=_time.time(),
    )
    await p.open_position(trade)

    data = p.to_dict()
    assert data["cash"] == pytest.approx(500 - 21.0 - 0.10, abs=0.01)
    assert "polymarket:tok1:poly_mm" in data["positions"]

    # Restore into a new portfolio
    p2 = Portfolio(capital_usd=500)
    p2.restore_from(data)
    assert p2.cash == pytest.approx(data["cash"], abs=0.01)
    assert len(p2.positions) == 1
    pos = list(p2.positions.values())[0]
    assert pos.avg_price == 0.50
    assert pos.size == 21.0


# --- Engine handle_fill ---

@pytest.mark.asyncio
async def test_handle_fill_reconciles_pending():
    """handle_fill matches fill to pending order and updates portfolio."""
    from core.portfolio import Portfolio
    from core.risk_manager import RiskConfig, RiskManager
    from core.engine import Engine

    portfolio = Portfolio(capital_usd=500)
    rm = RiskManager(portfolio, RiskConfig())
    engine = Engine(portfolio, rm, {"general": {"mode": "live"}})

    sig = Signal(
        strategy="poly_mm", market_id="m1", symbol="Yes",
        direction="buy", size_usd=21.0, price=0.50,
        metadata={"token_id": "tok1", "platform": "polymarket"},
    )

    # Register pending order
    engine._register_pending(sig, {"order_id": "ord_123", "token_id": "tok1"})
    assert "ord_123" in engine._pending_orders

    # Simulate fill event (we are the maker)
    fill_data = {
        "event_type": "trade",
        "asset_id": "tok1",
        "status": "MATCHED",
        "maker_orders": [
            {"order_id": "ord_123", "matched_amount": "42.0", "price": "0.50"},
        ],
        "taker_order_id": "taker_xyz",
        "side": "SELL",
        "size": "42.0",
    }
    await engine.handle_fill(fill_data)

    # Pending order should be consumed
    assert "ord_123" not in engine._pending_orders
    # Portfolio should have an open position
    assert len(portfolio.positions) == 1
