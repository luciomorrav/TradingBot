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

def test_track_order_and_fill_updates_inventory():
    """Paper mode: track_order + on_fill should update inventory correctly."""
    from strategies.poly_market_maker import MarketState, PolyMarketMaker
    from core.portfolio import Portfolio
    from core.risk_manager import RiskConfig, RiskManager
    from unittest.mock import MagicMock

    portfolio = Portfolio(capital_usd=500)
    rm = RiskManager(portfolio, RiskConfig())
    client = MagicMock()

    mm = PolyMarketMaker("test", portfolio, rm, {"market_maker": {}}, client)
    mm.market_states["t1"] = MarketState(token_id="t1", market_id="m1", outcome="YES")

    # Paper mode flow: track_order registers, on_fill pops and updates inventory
    mm.track_order("order_1", "t1", "buy", price=0.50, size=100, size_usd=50)
    assert "order_1" in mm._active_orders
    mm.on_fill("order_1", filled_size=100.0, filled_price=0.50)
    assert mm.market_states["t1"].inventory == 100.0
    assert "order_1" not in mm._active_orders

    mm.track_order("order_2", "t1", "sell", price=0.50, size=60, size_usd=30)
    mm.on_fill("order_2", filled_size=60.0, filled_price=0.50)
    assert mm.market_states["t1"].inventory == 40.0


def test_on_fill_partial_keeps_order():
    """Partial fills should update inventory but keep order until fully filled."""
    from strategies.poly_market_maker import MarketState, PolyMarketMaker
    from core.portfolio import Portfolio
    from core.risk_manager import RiskConfig, RiskManager
    from unittest.mock import MagicMock

    portfolio = Portfolio(capital_usd=500)
    rm = RiskManager(portfolio, RiskConfig())
    client = MagicMock()

    mm = PolyMarketMaker("test", portfolio, rm, {"market_maker": {}}, client)
    mm.market_states["t1"] = MarketState(token_id="t1", market_id="m1", outcome="YES")

    # Track order: 100 shares
    mm.track_order("ord_1", "t1", "buy", price=0.50, size=100, size_usd=50)

    # First partial fill: 40 of 100 shares
    mm.on_fill("ord_1", filled_size=40.0, filled_price=0.50)
    assert mm.market_states["t1"].inventory == 40.0
    assert "ord_1" in mm._active_orders  # still tracked

    # Second partial fill: 30 more shares (70 total < 99%)
    mm.on_fill("ord_1", filled_size=30.0, filled_price=0.50)
    assert mm.market_states["t1"].inventory == 70.0
    assert "ord_1" in mm._active_orders  # still tracked

    # Final fill: 30 more shares (100 total >= 99%)
    mm.on_fill("ord_1", filled_size=30.0, filled_price=0.50)
    assert mm.market_states["t1"].inventory == 100.0
    assert "ord_1" not in mm._active_orders  # now removed


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
async def test_validate_markets_soft_skips_tight_spread():
    """Markets with tight spread get 30min cooldown (not permanently removed)."""
    from connectors.polymarket_client import OrderBook, OrderBookLevel
    from strategies.poly_market_maker import MarketState
    from unittest.mock import AsyncMock

    mm = _make_mm()
    mm.client.subscribe_market = AsyncMock()

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

    # Both remain (soft-skip, not permanent drop)
    assert "t_wide" in mm.market_states
    assert "t_tight" in mm.market_states
    # Tight-spread token gets future cooldown (last_quote_time set ~30 min in future)
    assert mm.market_states["t_tight"].last_quote_time > time.time() + 1700


@pytest.mark.asyncio
async def test_two_sided_quoting_live_mode():
    """Live mode should generate both BUY + SELL signals when inventory exists."""
    from connectors.polymarket_client import OrderBook, OrderBookLevel
    from strategies.poly_market_maker import MarketState, PolyMarketMaker
    from core.portfolio import Portfolio, Position, Platform, Side
    from core.risk_manager import RiskConfig, RiskManager
    from unittest.mock import MagicMock

    portfolio = Portfolio(capital_usd=500)
    rm = RiskManager(portfolio, RiskConfig())
    client = MagicMock()

    mm = PolyMarketMaker("test", portfolio, rm, {"market_maker": {}}, client)
    mm._live_mode = True

    state = MarketState(token_id="t1", market_id="m1", outcome="YES", inventory=50)
    for _ in range(10):
        state.mid_prices.append(0.50)
    mm.market_states["t1"] = state

    # Must have a matching portfolio position so inventory resync doesn't clear it
    portfolio.positions["polymarket:t1:test"] = Position(
        platform=Platform.POLYMARKET, market_id="t1", symbol="YES",
        side=Side.BUY, avg_price=0.50, size=25.0, strategy="test",
    )

    book = OrderBook("m1", "t1",
                     bids=[OrderBookLevel(0.44, 100)],
                     asks=[OrderBookLevel(0.56, 100)])
    client.get_order_book = lambda tid: book

    signals = await mm.evaluate()
    directions = {s.direction for s in signals}
    assert "buy" in directions, "Should have a BUY signal"
    assert "sell" in directions, "Should have a SELL signal"


@pytest.mark.asyncio
async def test_one_sided_quoting_paper_mode():
    """Paper mode should generate only ONE signal per token per cycle."""
    from connectors.polymarket_client import OrderBook, OrderBookLevel
    from strategies.poly_market_maker import MarketState, PolyMarketMaker
    from core.portfolio import Portfolio
    from core.risk_manager import RiskConfig, RiskManager
    from unittest.mock import MagicMock

    portfolio = Portfolio(capital_usd=500)
    rm = RiskManager(portfolio, RiskConfig())
    client = MagicMock()

    mm = PolyMarketMaker("test", portfolio, rm, {"market_maker": {}}, client)
    mm._live_mode = False

    state = MarketState(token_id="t1", market_id="m1", outcome="YES", inventory=50)
    for _ in range(10):
        state.mid_prices.append(0.50)
    mm.market_states["t1"] = state

    book = OrderBook("m1", "t1",
                     bids=[OrderBookLevel(0.44, 100)],
                     asks=[OrderBookLevel(0.56, 100)])
    client.get_order_book = lambda tid: book

    signals = await mm.evaluate()
    # Paper mode: only one side per token
    assert len(signals) == 1


def test_has_live_orders_side_aware():
    """_has_live_orders with side param should only match that side."""
    from strategies.poly_market_maker import PolyMarketMaker, LiveOrder
    from core.portfolio import Portfolio
    from core.risk_manager import RiskConfig, RiskManager
    from unittest.mock import MagicMock

    portfolio = Portfolio(capital_usd=500)
    rm = RiskManager(portfolio, RiskConfig())
    client = MagicMock()

    mm = PolyMarketMaker("test", portfolio, rm, {"market_maker": {}}, client)

    # Track a BUY order
    mm._active_orders["ord1"] = LiveOrder(
        order_id="ord1", token_id="t1", side="buy",
        price=0.48, size=10, size_usd=5, placed_at=time.time(),
    )

    # No side filter → True (any order exists)
    assert mm._has_live_orders("t1") is True
    # Side "buy" → True
    assert mm._has_live_orders("t1", "buy") is True
    # Side "sell" → False (only buy exists)
    assert mm._has_live_orders("t1", "sell") is False


@pytest.mark.asyncio
async def test_signal_cap_per_cycle():
    """Max signals per cycle should be capped at MAX_SIGNALS_PER_CYCLE."""
    from connectors.polymarket_client import OrderBook, OrderBookLevel
    from strategies.poly_market_maker import MarketState, PolyMarketMaker
    from core.portfolio import Portfolio, Position, Platform, Side
    from core.risk_manager import RiskConfig, RiskManager
    from unittest.mock import MagicMock

    portfolio = Portfolio(capital_usd=5000)
    rm = RiskManager(portfolio, RiskConfig())
    client = MagicMock()

    mm = PolyMarketMaker("test", portfolio, rm, {"market_maker": {}}, client)
    mm._live_mode = True

    # Add 10 tokens — all with inventory for two-sided quoting
    for i in range(10):
        tid = f"t{i}"
        state = MarketState(
            token_id=tid, market_id=f"m{i}", outcome="YES", inventory=50,
        )
        for _ in range(10):
            state.mid_prices.append(0.50)
        mm.market_states[tid] = state
        # Must have matching portfolio position (inventory resync)
        portfolio.positions[f"polymarket:{tid}:test"] = Position(
            platform=Platform.POLYMARKET, market_id=tid, symbol="YES",
            side=Side.BUY, avg_price=0.50, size=25.0, strategy="test",
        )

    book = OrderBook("m0", "t0",
                     bids=[OrderBookLevel(0.44, 100)],
                     asks=[OrderBookLevel(0.56, 100)])
    client.get_order_book = lambda tid: book

    signals = await mm.evaluate()
    assert len(signals) <= 6, f"Expected max 6 signals, got {len(signals)}"


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
    assert pos.entry_time == pytest.approx(trade.timestamp, abs=1.0)  # entry_time preserved


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

    # Simulate full fill event (we are the maker, 42 shares = $21 / $0.50)
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

    # Pending order should be consumed (42 shares >= 42 shares * 0.99)
    assert "ord_123" not in engine._pending_orders
    # Portfolio should have an open position
    assert len(portfolio.positions) == 1


@pytest.mark.asyncio
async def test_handle_fill_dedup_ignores_mined():
    """MINED/CONFIRMED events for the same fill should be ignored (only MATCHED processed)."""
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
    engine._register_pending(sig, {"order_id": "ord_456", "token_id": "tok1"})

    base_fill = {
        "event_type": "trade",
        "asset_id": "tok1",
        "maker_orders": [
            {"order_id": "ord_456", "matched_amount": "42.0", "price": "0.50"},
        ],
        "taker_order_id": "taker_xyz",
        "side": "SELL",
        "size": "42.0",
    }

    # First: MATCHED — should process
    await engine.handle_fill({**base_fill, "status": "MATCHED"})
    assert len(portfolio.positions) == 1

    # Second: MINED — should be ignored (not MATCHED)
    await engine.handle_fill({**base_fill, "status": "MINED"})
    # Portfolio still has exactly 1 position (not doubled)
    pos = list(portfolio.positions.values())[0]
    assert pos.size == pytest.approx(21.0, abs=0.5)


@pytest.mark.asyncio
async def test_handle_fill_partial_keeps_pending():
    """A partial fill should keep the pending order until fully filled."""
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
    engine._register_pending(sig, {"order_id": "ord_789", "token_id": "tok1"})

    # First partial fill: 20 of 42 shares
    fill_1 = {
        "event_type": "trade",
        "asset_id": "tok1",
        "status": "MATCHED",
        "maker_orders": [
            {"order_id": "ord_789", "matched_amount": "20.0", "price": "0.50"},
        ],
        "taker_order_id": "taker_a",
        "side": "SELL",
        "size": "20.0",
    }
    await engine.handle_fill(fill_1)
    # Pending order should still exist (20 < 42 * 0.99)
    assert "ord_789" in engine._pending_orders

    # Second partial fill: remaining 22 shares
    fill_2 = {
        "event_type": "trade",
        "asset_id": "tok1",
        "status": "MATCHED",
        "maker_orders": [
            {"order_id": "ord_789", "matched_amount": "22.0", "price": "0.50"},
        ],
        "taker_order_id": "taker_b",
        "side": "SELL",
        "size": "22.0",
    }
    await engine.handle_fill(fill_2)
    # Now fully filled — should be removed
    assert "ord_789" not in engine._pending_orders


@pytest.mark.asyncio
async def test_paper_execute_no_cash_overdraw():
    """Multiple BUY signals in one cycle must not overdraw cash."""
    from core.portfolio import Portfolio
    from core.risk_manager import RiskConfig, RiskManager
    from core.engine import Engine

    portfolio = Portfolio(capital_usd=30.0)  # tight cash
    rm = RiskManager(portfolio, RiskConfig())
    engine = Engine(portfolio, rm, {"general": {"mode": "paper"},
                                    "polymarket": {"paper_fee_rate": 0.005}})

    # Three BUY signals, each wanting $21 — total $63 > $30 available
    for i in range(3):
        sig = Signal(
            strategy="poly_mm", market_id=f"m{i}", symbol="Yes",
            direction="buy", size_usd=21.0, price=0.50,
            metadata={"token_id": f"tok{i}", "platform": "polymarket", "fee": 0.0},
        )
        await engine._paper_execute(sig)

    # Cash must never go meaningfully negative (allow floating point epsilon)
    assert portfolio.cash >= -0.01, f"Cash went negative: {portfolio.cash}"


@pytest.mark.asyncio
async def test_close_position_partial_pnl():
    """Partial close should calculate PnL correctly using shares, not USD notional."""
    from core.portfolio import Portfolio, Platform, Side, Trade
    import time as _time

    p = Portfolio(capital_usd=500)

    # Open: buy 100 shares at $0.50 → $50 notional
    buy = Trade(
        trade_id="buy1", platform=Platform.POLYMARKET, market_id="tok1",
        symbol="Yes", side=Side.BUY, price=0.50, size=50.0, fee=0.0,
        strategy="test", timestamp=_time.time(),
    )
    await p.open_position(buy)
    assert p.cash == pytest.approx(450.0, abs=0.01)

    # Partial close: sell 50 shares at $0.60 → $30 notional
    sell = Trade(
        trade_id="sell1", platform=Platform.POLYMARKET, market_id="tok1",
        symbol="Yes", side=Side.SELL, price=0.60, size=30.0, fee=0.0,
        strategy="test", timestamp=_time.time(),
    )
    pnl = await p.close_position(sell)

    # 50 shares * (0.60 - 0.50) = $5.00 PnL
    assert pnl == pytest.approx(5.0, abs=0.01)
    # Cash: 450 + 50 shares * 0.60 = 450 + 30 = 480
    assert p.cash == pytest.approx(480.0, abs=0.01)
    # Remaining position: 50 shares at $0.50 = $25 cost basis
    assert len(p.positions) == 1
    pos = list(p.positions.values())[0]
    assert pos.size == pytest.approx(25.0, abs=0.5)


@pytest.mark.asyncio
async def test_paper_drawdown_triggers_kill_switch():
    """Paper mode should trigger kill switch when drawdown exceeds limit."""
    from core.portfolio import Portfolio
    from core.risk_manager import RiskConfig, RiskManager
    from core.engine import Engine

    portfolio = Portfolio(capital_usd=100.0)
    rc = RiskConfig(max_daily_drawdown_pct=5.0)
    rm = RiskManager(portfolio, rc)

    engine = Engine(portfolio, rm, {"general": {"mode": "paper"},
                                    "polymarket": {"paper_fee_rate": 0.0}})

    # Simulate loss: reduce cash to trigger 6% drawdown
    portfolio.cash = 94.0  # equity = 94, drawdown = 6% > 5%

    sig = Signal(
        strategy="poly_mm", market_id="m1", symbol="Yes",
        direction="buy", size_usd=10.0, price=0.50,
        metadata={"token_id": "tok1", "platform": "polymarket", "fee": 0.0},
    )
    await engine._paper_execute(sig)

    # Kill switch should have triggered — signal should be blocked
    assert rm.killed is True
    # No position should have been opened
    assert len(portfolio.positions) == 0


@pytest.mark.asyncio
async def test_paper_sell_no_position_skipped():
    """SELL signal in paper mode with no matching position should be skipped (no short)."""
    from core.portfolio import Portfolio
    from core.risk_manager import RiskConfig, RiskManager
    from core.engine import Engine

    portfolio = Portfolio(capital_usd=100.0)
    rm = RiskManager(portfolio, RiskConfig())
    engine = Engine(portfolio, rm, {"general": {"mode": "paper"},
                                    "polymarket": {"paper_fee_rate": 0.0}})

    sig = Signal(
        strategy="poly_mm", market_id="m1", symbol="Yes",
        direction="sell", size_usd=10.0, price=0.50,
        metadata={"token_id": "tok1", "platform": "polymarket", "fee": 0.0},
    )
    await engine._paper_execute(sig)

    # No position should be created (no short)
    assert len(portfolio.positions) == 0
    # Cash should be unchanged
    assert portfolio.cash == 100.0


# --- Reconciliation tests ---


@pytest.mark.asyncio
async def test_reconciliation_alerts_on_cash_desync():
    """Reconciliation should alert when exchange balance differs from local by > $30."""
    from core.portfolio import Portfolio
    from core.risk_manager import RiskConfig, RiskManager
    from core.engine import Engine
    from unittest.mock import AsyncMock, MagicMock

    portfolio = Portfolio(capital_usd=340.0)
    rm = RiskManager(portfolio, RiskConfig())
    engine = Engine(portfolio, rm, {"general": {"mode": "live"}})

    # Mock poly_client with balance desync ($340 local vs $300 exchange = $40 delta)
    mock_client = MagicMock()
    mock_client.get_exchange_balance = AsyncMock(return_value=300.0)
    mock_client.get_exchange_orders = AsyncMock(return_value=[])
    mock_client.get_exchange_positions = AsyncMock(return_value=[])
    engine.poly_client = mock_client

    alerts = []
    engine.notify_callback = AsyncMock(side_effect=lambda msg: alerts.append(msg))

    await engine._run_reconciliation()

    assert len(alerts) == 1
    assert "Cash desync" in alerts[0]
    assert "300.00" in alerts[0]


@pytest.mark.asyncio
async def test_reconciliation_no_alert_when_matched():
    """Reconciliation should not alert when states are within thresholds."""
    from core.portfolio import Portfolio, Platform, Side, Trade
    from core.risk_manager import RiskConfig, RiskManager
    from core.engine import Engine
    from unittest.mock import AsyncMock, MagicMock

    portfolio = Portfolio(capital_usd=340.0)
    # Add a position so local and exchange match
    trade = Trade(
        trade_id="t1", platform=Platform.POLYMARKET, market_id="tok_abc",
        symbol="Yes", side=Side.BUY, price=0.50, size=10.0, fee=0.0,
        strategy="poly_mm",
    )
    await portfolio.open_position(trade)
    rm = RiskManager(portfolio, RiskConfig())
    engine = Engine(portfolio, rm, {"general": {"mode": "live"}})

    # Mock poly_client — everything matches within threshold
    mock_client = MagicMock()
    mock_client.get_exchange_balance = AsyncMock(return_value=331.0)  # $330 local, $331 exchange = $1 delta
    mock_client.get_exchange_orders = AsyncMock(return_value=[])
    mock_client.get_exchange_positions = AsyncMock(return_value=[
        {"asset": "tok_abc", "size": "20.0"},  # 20 shares matches local (10/0.5=20)
    ])
    engine.poly_client = mock_client

    alerts = []
    engine.notify_callback = AsyncMock(side_effect=lambda msg: alerts.append(msg))

    await engine._run_reconciliation()

    # No alerts — everything within threshold
    assert len(alerts) == 0


@pytest.mark.asyncio
async def test_reconciliation_alerts_on_position_desync():
    """Reconciliation should alert when exchange has positions not tracked locally."""
    from core.portfolio import Portfolio
    from core.risk_manager import RiskConfig, RiskManager
    from core.engine import Engine
    from unittest.mock import AsyncMock, MagicMock

    portfolio = Portfolio(capital_usd=340.0)
    rm = RiskManager(portfolio, RiskConfig())
    engine = Engine(portfolio, rm, {"general": {"mode": "live"}})

    # Exchange has 2 positions, bot has 0
    mock_client = MagicMock()
    mock_client.get_exchange_balance = AsyncMock(return_value=340.0)
    mock_client.get_exchange_orders = AsyncMock(return_value=[])
    mock_client.get_exchange_positions = AsyncMock(return_value=[
        {"asset": "tok_orphan1", "size": "10.0"},
        {"asset": "tok_orphan2", "size": "5.0"},
    ])
    engine.poly_client = mock_client

    alerts = []
    engine.notify_callback = AsyncMock(side_effect=lambda msg: alerts.append(msg))

    await engine._run_reconciliation()

    assert len(alerts) == 1
    assert "exchange not in bot: 2" in alerts[0]


# --- LLM client tests ---


@pytest.mark.asyncio
async def test_llm_parse_valid_response():
    """LLMClient should parse valid JSON response from the model."""
    from connectors.llm_client import LLMClient
    from unittest.mock import MagicMock

    client = LLMClient({"model": "claude-haiku-4-5-20251001", "max_cost_per_day": 1.0})

    # Mock anthropic response
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text='{"probability": 0.72, "confidence": 0.85, "reasoning": "Strong evidence."}')]
    mock_resp.usage.input_tokens = 500
    mock_resp.usage.output_tokens = 50

    mock_anthropic_client = MagicMock()
    mock_anthropic_client.messages.create.return_value = mock_resp
    client._client = mock_anthropic_client

    result = await client.estimate_probability("Will X happen?", "Yes", 0.50, ["Headline 1"])

    assert result is not None
    assert result["probability"] == 0.72
    assert result["confidence"] == 0.85
    assert "Strong evidence" in result["reasoning"]
    assert result["cost_usd"] > 0


@pytest.mark.asyncio
async def test_llm_budget_cap_reached():
    """LLMClient should return None when daily cost exceeds budget."""
    from connectors.llm_client import LLMClient

    client = LLMClient({"max_cost_per_day": 0.001})
    client._daily_cost = 0.002  # already over budget

    result = await client.estimate_probability("Will X happen?", "Yes", 0.50, ["Headline 1"])

    assert result is None


# --- News Edge strategy tests ---


def _make_news_edge(shadow=True):
    """Helper: create a PolyNewsEdge with mocked dependencies."""
    from strategies.poly_news_edge import PolyNewsEdge
    from core.portfolio import Portfolio
    from core.risk_manager import RiskConfig, RiskManager
    from unittest.mock import MagicMock, AsyncMock

    portfolio = Portfolio(capital_usd=340)
    rm = RiskManager(portfolio, RiskConfig())
    poly_client = MagicMock()
    poly_client.fetch_active_markets = AsyncMock(return_value=[])
    poly_client.subscribe_market = AsyncMock()
    llm_client = MagicMock()
    news_scraper = MagicMock()

    config = {
        "news_edge": {
            "enabled": True,
            "shadow_mode": shadow,
            "edge_threshold": 0.12,
            "min_confidence": 0.65,
            "max_position_per_market": 10,
            "strategy_cap_pct": 10.0,
            "cooldown_hours": 4,
            "take_profit_pct": 0.20,
            "stop_loss_pct": 0.15,
            "max_hold_hours": 48,
        },
        "market_maker": {},
    }
    ne = PolyNewsEdge("news_edge", portfolio, rm, config, poly_client, llm_client, news_scraper)
    return ne, portfolio, llm_client, news_scraper


@pytest.mark.asyncio
async def test_news_edge_buy_yes_signal():
    """LLM prob=0.80, market=0.50 → should generate BUY Yes signal (edge=+0.30)."""
    from connectors.polymarket_client import Market
    from unittest.mock import AsyncMock

    ne, portfolio, llm, scraper = _make_news_edge(shadow=False)

    # Mock news
    scraper.fetch_news = AsyncMock(return_value=([{"title": "Breaking news"}], "hash123"))

    # Mock LLM: prob=0.80, conf=0.90 → edge=+0.30 (Yes underpriced)
    llm.estimate_probability = AsyncMock(return_value={
        "probability": 0.80, "confidence": 0.90, "reasoning": "Strong signal", "cost_usd": 0.001,
    })

    market = Market(
        id="m1", question="Will X happen?", slug="will-x", active=True,
        end_date="2026-06-01T00:00:00Z",
        tokens=[
            {"token_id": "tok_yes", "outcome": "Yes", "price": 0.50},
            {"token_id": "tok_no", "outcome": "No", "price": 0.50},
        ],
        volume=5000, liquidity=1000,
    )
    ne._markets = [market]
    ne._last_refresh = time.time()  # skip refresh

    signals = await ne.evaluate()

    assert len(signals) == 1
    sig = signals[0]
    assert sig.direction == "buy"
    assert sig.market_id == "tok_yes"  # BUY Yes (underpriced)
    assert sig.metadata.get("edge") == 0.30
    assert "shadow" not in sig.metadata  # not shadow mode


@pytest.mark.asyncio
async def test_news_edge_buy_no_signal():
    """LLM prob=0.20, market=0.50 → should generate BUY No signal (edge=-0.30)."""
    from connectors.polymarket_client import Market
    from unittest.mock import AsyncMock

    ne, portfolio, llm, scraper = _make_news_edge(shadow=False)

    scraper.fetch_news = AsyncMock(return_value=([{"title": "Bad news"}], "hash456"))
    llm.estimate_probability = AsyncMock(return_value={
        "probability": 0.20, "confidence": 0.85, "reasoning": "Unlikely", "cost_usd": 0.001,
    })

    market = Market(
        id="m2", question="Will Y happen?", slug="will-y", active=True,
        end_date="2026-06-01T00:00:00Z",
        tokens=[
            {"token_id": "tok_yes2", "outcome": "Yes", "price": 0.50},
            {"token_id": "tok_no2", "outcome": "No", "price": 0.50},
        ],
        volume=5000, liquidity=1000,
    )
    ne._markets = [market]
    ne._last_refresh = time.time()

    signals = await ne.evaluate()

    assert len(signals) == 1
    sig = signals[0]
    assert sig.direction == "buy"
    assert sig.market_id == "tok_no2"  # BUY No (Yes is overpriced)


@pytest.mark.asyncio
async def test_news_edge_cooldown_skip():
    """Same market analyzed within cooldown should be skipped."""
    from connectors.polymarket_client import Market
    from unittest.mock import AsyncMock

    ne, portfolio, llm, scraper = _make_news_edge()

    # Mark market as recently analyzed (with known hash)
    ne._analyzed["m3"] = (time.time(), "somehash")

    # News fetch now happens before cooldown check — mock it
    # Return NEW headlines but different hash to exercise cooldown path
    scraper.fetch_news = AsyncMock(return_value=([{"title": "New headline"}], "newhash"))

    market = Market(
        id="m3", question="Will Z happen?", slug="will-z", active=True,
        end_date="2026-06-01T00:00:00Z",
        tokens=[
            {"token_id": "tok_yes3", "outcome": "Yes", "price": 0.50},
            {"token_id": "tok_no3", "outcome": "No", "price": 0.50},
        ],
        volume=5000, liquidity=1000,
    )
    ne._markets = [market]
    ne._last_refresh = time.time()

    signals = await ne.evaluate()

    # Should be skipped by cooldown (new news but 4h not elapsed) — no LLM call
    assert len(signals) == 0
    llm.estimate_probability.assert_not_called()


@pytest.mark.asyncio
async def test_news_edge_tp_exit():
    """Position at +25% PnL should generate a SELL signal (take profit)."""
    from core.portfolio import Platform, Side, Trade

    ne, portfolio, llm, scraper = _make_news_edge()

    # Open a position manually
    trade = Trade(
        trade_id="t1", platform=Platform.POLYMARKET, market_id="tok_tp",
        symbol="Yes — Test", side=Side.BUY, price=0.40, size=10.0, fee=0.0,
        strategy="news_edge", timestamp=time.time() - 3600,
    )
    await portfolio.open_position(trade)

    # Simulate price increase → +25% PnL
    pos = list(portfolio.positions.values())[0]
    pos.current_price = 0.50  # (0.50 - 0.40) / 0.40 = +25%

    ne._markets = []
    ne._last_refresh = time.time()

    signals = await ne.evaluate()

    assert len(signals) == 1
    sig = signals[0]
    assert sig.direction == "sell"
    assert sig.metadata.get("reason") == "TP"


@pytest.mark.asyncio
async def test_news_edge_shadow_no_execute():
    """Shadow signal should have shadow=True in metadata, engine skips execution."""
    from connectors.polymarket_client import Market
    from core.engine import Engine
    from core.risk_manager import RiskConfig, RiskManager
    from unittest.mock import AsyncMock

    ne, portfolio, llm, scraper = _make_news_edge(shadow=True)

    scraper.fetch_news = AsyncMock(return_value=([{"title": "News"}], "hash789"))
    llm.estimate_probability = AsyncMock(return_value={
        "probability": 0.80, "confidence": 0.90, "reasoning": "Test", "cost_usd": 0.001,
    })

    market = Market(
        id="m4", question="Shadow test?", slug="shadow", active=True,
        end_date="2026-06-01T00:00:00Z",
        tokens=[
            {"token_id": "tok_shadow", "outcome": "Yes", "price": 0.50},
            {"token_id": "tok_shadow_no", "outcome": "No", "price": 0.50},
        ],
        volume=5000, liquidity=1000,
    )
    ne._markets = [market]
    ne._last_refresh = time.time()

    signals = await ne.evaluate()
    assert len(signals) == 1
    assert signals[0].metadata.get("shadow") is True

    # Engine should not execute shadow signals
    rm = RiskManager(portfolio, RiskConfig())
    engine = Engine(portfolio, rm, {"general": {"mode": "paper"}, "polymarket": {}})
    notifications = []
    engine.notify_callback = AsyncMock(side_effect=lambda msg: notifications.append(msg))

    await engine._process_signal(signals[0])

    # No position opened (shadow = no execution)
    assert len(portfolio.positions) == 0
    # But notification was sent
    assert len(notifications) == 1
    assert "Shadow" in notifications[0]


# --- Regression tests (audit fixes) ---


@pytest.mark.asyncio
async def test_news_edge_tp_full_close():
    """TP exit at higher price should close the FULL position (no dust left).

    Regression: size_usd=pos.size at different price left partial position.
    Fix: size_usd = shares * current_price so close_position() recovers exact share count.
    """
    from core.portfolio import Platform, Side, Trade

    ne, portfolio, llm, scraper = _make_news_edge()

    trade = Trade(
        trade_id="t_full", platform=Platform.POLYMARKET, market_id="tok_full",
        symbol="Yes — Full", side=Side.BUY, price=0.40, size=10.0, fee=0.0,
        strategy="news_edge", timestamp=time.time() - 3600,
    )
    await portfolio.open_position(trade)

    pos = list(portfolio.positions.values())[0]
    pos.current_price = 0.50  # +25% → TP

    ne._markets = []
    ne._last_refresh = time.time()

    signals = await ne.evaluate()
    assert len(signals) == 1
    sig = signals[0]

    # size_usd should be shares * price, not pos.size
    shares = 10.0 / 0.40  # 25 shares
    expected_size = shares * 0.50  # $12.50
    assert abs(sig.size_usd - expected_size) < 0.01

    # Simulate close via portfolio
    close_trade = Trade(
        trade_id="t_close", platform=Platform.POLYMARKET, market_id="tok_full",
        symbol="Yes — Full", side=Side.SELL, price=sig.price, size=sig.size_usd,
        fee=0.0, strategy="news_edge",
    )
    await portfolio.close_position(close_trade)

    # Position should be fully closed (no dust)
    assert len(portfolio.positions) == 0


@pytest.mark.asyncio
async def test_sell_exit_bypasses_risk_manager():
    """SELL signals should not be blocked by risk manager (cash/exposure checks).

    Regression: run_cycle() applied check_can_trade() to SELL, blocking SL exits.
    """
    from core.portfolio import Platform, Side, Trade
    from core.risk_manager import RiskConfig, RiskManager

    ne, portfolio, llm, scraper = _make_news_edge()

    # Drain all cash — risk manager would block any BUY
    portfolio.cash = 0.0

    trade = Trade(
        trade_id="t_sl", platform=Platform.POLYMARKET, market_id="tok_sl",
        symbol="Yes — SL", side=Side.BUY, price=0.50, size=10.0, fee=0.0,
        strategy="news_edge", timestamp=time.time() - 3600,
    )
    await portfolio.open_position(trade)

    pos = list(portfolio.positions.values())[0]
    pos.current_price = 0.40  # -20% → SL

    ne._markets = []
    ne._last_refresh = time.time()

    # run_cycle() applies risk checks — SELL must still pass
    signals = await ne.run_cycle()

    assert len(signals) == 1
    assert signals[0].direction == "sell"
    assert signals[0].metadata.get("reason") == "SL"


@pytest.mark.asyncio
async def test_news_edge_skip_exit_no_price():
    """Position with current_price=0 (no WS update yet) should NOT trigger false SL.

    Regression: current_price=0 after restart → PnL=-100% → instant SL.
    """
    from core.portfolio import Platform, Side, Trade

    ne, portfolio, llm, scraper = _make_news_edge()

    trade = Trade(
        trade_id="t_no_price", platform=Platform.POLYMARKET, market_id="tok_np",
        symbol="Yes — NP", side=Side.BUY, price=0.50, size=10.0, fee=0.0,
        strategy="news_edge", timestamp=time.time() - 3600,
    )
    await portfolio.open_position(trade)

    # current_price stays at 0.0 (no WS update yet)
    pos = list(portfolio.positions.values())[0]
    assert pos.current_price == 0.0

    ne._markets = []
    ne._last_refresh = time.time()

    signals = await ne.evaluate()

    # Should NOT generate any exit signal
    assert len(signals) == 0


@pytest.mark.asyncio
async def test_news_edge_no_token_uses_complement_price():
    """BUY No signal should use 1 - yes_price, not the No token's stale price.

    Regression: No token's API price was 0.0005 (thin book) while Yes=0.96,
    producing a signal @ 0.0005 instead of @ 0.04.
    """
    from connectors.polymarket_client import Market
    from unittest.mock import AsyncMock

    ne, portfolio, llm, scraper = _make_news_edge(shadow=False)

    scraper.fetch_news = AsyncMock(return_value=([{"title": "News"}], "hash_no"))
    llm.estimate_probability = AsyncMock(return_value={
        "probability": 0.05, "confidence": 0.80, "reasoning": "Very unlikely", "cost_usd": 0.001,
    })

    market = Market(
        id="m_no", question="Will rare event happen?", slug="rare-event", active=True,
        end_date="2026-06-01T00:00:00Z",
        tokens=[
            {"token_id": "tok_yes_r", "outcome": "Yes", "price": 0.80},
            # No token has stale/wrong price from thin order book
            {"token_id": "tok_no_r", "outcome": "No", "price": 0.0005},
        ],
        volume=5000, liquidity=1000,
    )
    ne._markets = [market]
    ne._last_refresh = time.time()

    signals = await ne.evaluate()

    assert len(signals) == 1
    sig = signals[0]
    assert sig.market_id == "tok_no_r"  # BUY No
    # Price should be complement (1 - 0.80 = 0.20), NOT 0.0005
    assert abs(sig.price - 0.20) < 0.01


@pytest.mark.asyncio
async def test_news_edge_skip_low_price():
    """Skip signal when buy_price is below minimum threshold (0.02).

    E.g. Yes=0.99 → No complement = 0.01 → too low to be a real trade.
    """
    from connectors.polymarket_client import Market
    from unittest.mock import AsyncMock

    ne, portfolio, llm, scraper = _make_news_edge(shadow=False)

    scraper.fetch_news = AsyncMock(return_value=([{"title": "News"}], "hash_low"))
    llm.estimate_probability = AsyncMock(return_value={
        "probability": 0.02, "confidence": 0.85, "reasoning": "Almost certain", "cost_usd": 0.001,
    })

    market = Market(
        id="m_low", question="Will certain event happen?", slug="certain", active=True,
        end_date="2026-06-01T00:00:00Z",
        tokens=[
            {"token_id": "tok_yes_c", "outcome": "Yes", "price": 0.99},
            {"token_id": "tok_no_c", "outcome": "No", "price": 0.01},
        ],
        volume=5000, liquidity=1000,
    )
    ne._markets = [market]
    ne._last_refresh = time.time()

    signals = await ne.evaluate()

    # Should skip — complement price 0.01 < 0.02 minimum
    assert len(signals) == 0
