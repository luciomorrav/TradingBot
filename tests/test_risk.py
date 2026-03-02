"""Tests for risk manager — kill switch, sizing, cooldown, execution degradation."""
import asyncio
import time

import pytest

from core.portfolio import Platform, Portfolio, Side, Trade
from core.risk_manager import ExecutionMetrics, RiskConfig, RiskManager


@pytest.fixture
def portfolio():
    return Portfolio(capital_usd=500)


@pytest.fixture
def risk_config():
    return RiskConfig(
        max_daily_drawdown_pct=5.0,
        max_position_pct=20.0,
        max_total_exposure_pct=60.0,
        consecutive_loss_cooldown=3,
        cooldown_minutes=1,
        min_fill_rate=0.80,
        max_slippage_pct=1.0,
        max_latency_seconds=5.0,
    )


@pytest.fixture
def rm(portfolio, risk_config):
    return RiskManager(portfolio, risk_config)


def _make_trade(strategy="test", price=0.5, size=50, side="buy", slippage=0.0, latency=10.0):
    return Trade(
        trade_id="t1", platform=Platform.POLYMARKET, market_id="m1",
        symbol="TEST", side=Side.BUY if side == "buy" else Side.SELL,
        price=price, size=size, slippage=slippage, strategy=strategy,
        timestamp=time.time(), latency_ms=latency,
    )


# --- Basic risk checks ---

@pytest.mark.asyncio
async def test_check_can_trade_ok(rm):
    ok, reason = await rm.check_can_trade("test", 50)
    assert ok, reason


@pytest.mark.asyncio
async def test_check_position_size_limit(rm):
    # max_position_pct=20% of $500 = $100
    ok, reason = await rm.check_can_trade("test", 150)
    assert not ok
    assert "max" in reason.lower() or "Position" in reason


@pytest.mark.asyncio
async def test_check_exposure_limit(rm, portfolio):
    # max_total_exposure=60% of $500 = $300
    # Simulate exposure by opening positions
    for i in range(4):
        t = _make_trade(size=80)
        t.trade_id = f"t{i}"
        await portfolio.open_position(t)

    # Now total exposure should be ~320, trying to add more should fail
    ok, reason = await rm.check_can_trade("test", 50)
    assert not ok
    assert "exposure" in reason.lower() or "cash" in reason.lower()


@pytest.mark.asyncio
async def test_check_cash_limit(rm, portfolio):
    # Cash starts at 500, open position that uses most of it
    t = _make_trade(size=490)
    await portfolio.open_position(t)

    ok, reason = await rm.check_can_trade("test", 50)
    assert not ok
    # Might hit exposure limit or cash limit first — either is a valid rejection
    assert any(w in reason.lower() for w in ("exposure", "cash", "insufficient"))


# --- Kill switch ---

@pytest.mark.asyncio
async def test_kill_switch_blocks_trading(rm):
    await rm.manual_kill()
    assert rm.killed
    ok, reason = await rm.check_can_trade("test", 10)
    assert not ok
    assert "kill" in reason.lower()


@pytest.mark.asyncio
async def test_resume_after_kill(rm):
    await rm.manual_kill()
    assert rm.killed
    rm.resume()
    assert not rm.killed
    ok, _ = await rm.check_can_trade("test", 10)
    assert ok


@pytest.mark.asyncio
async def test_kill_switch_on_drawdown(rm, portfolio):
    # Simulate drawdown > 5%: start equity = 500, lose $30 = 6%
    rm.daily_start_equity = 500
    portfolio.cash = 470  # simulate loss — equity drops to 470
    ok, reason = await rm.check_can_trade("test", 10)
    assert not ok
    assert rm.killed


# --- Consecutive loss cooldown ---

@pytest.mark.asyncio
async def test_cooldown_after_consecutive_losses(rm):
    for _ in range(3):
        trade = _make_trade()
        await rm.record_trade(trade, filled=True, pnl=-5.0)

    ok, reason = await rm.check_can_trade("test", 10)
    assert not ok
    assert "cooldown" in reason.lower()


@pytest.mark.asyncio
async def test_win_resets_consecutive_losses(rm):
    for _ in range(2):
        await rm.record_trade(_make_trade(), filled=True, pnl=-5.0)
    # Win should reset
    await rm.record_trade(_make_trade(), filled=True, pnl=10.0)
    assert rm.consecutive_losses.get("test", 0) == 0


# --- Dynamic sizing ---

def test_suggest_position_size_basic(rm):
    size = rm.suggest_position_size("test", 50)
    assert 0 < size <= 100  # max_position_pct=20% of $500


def test_suggest_position_size_high_volatility(rm):
    normal = rm.suggest_position_size("test", 50, volatility=0.0)
    high_vol = rm.suggest_position_size("test", 50, volatility=0.5)
    assert high_vol < normal  # high vol should reduce size


def test_suggest_position_size_low_fill_rate(rm):
    normal = rm.suggest_position_size("test", 50)

    # Simulate low fill rate
    metrics = ExecutionMetrics()
    for _ in range(15):
        metrics.record(filled=False, slippage=0, latency_ms=10)
    rm.execution_metrics["test"] = metrics

    degraded = rm.suggest_position_size("test", 50)
    assert degraded < normal


# --- Execution metrics ---

def test_execution_metrics_fill_rate():
    m = ExecutionMetrics()
    for _ in range(8):
        m.record(filled=True, slippage=0.001, latency_ms=50)
    for _ in range(2):
        m.record(filled=False, slippage=0, latency_ms=100)

    assert 0.79 < m.fill_rate < 0.81
    assert m.avg_slippage > 0
    assert m.avg_latency_ms > 0


@pytest.mark.asyncio
async def test_execution_degradation_blocks_trade(rm):
    metrics = ExecutionMetrics()
    # All trades have high slippage
    for _ in range(15):
        metrics.record(filled=True, slippage=2.0, latency_ms=50)
    rm.execution_metrics["test"] = metrics

    ok, reason = await rm.check_can_trade("test", 10)
    assert not ok
    assert "slippage" in reason.lower()


# --- Daily reset ---

def test_daily_reset(rm, portfolio):
    portfolio.cash = 450  # simulate loss
    rm.reset_daily()
    assert rm.daily_start_equity == portfolio.equity
    assert rm.daily_start_equity == 450  # verify cash change took effect


# --- Portfolio netting (opposite side = close) ---

@pytest.mark.asyncio
async def test_portfolio_netting_opposite_side_closes(portfolio):
    """Buying then selling same market should reduce position, not double it."""
    buy = _make_trade(strategy="mm", price=0.40, size=20, side="buy")
    buy.market_id = "token_123"
    await portfolio.open_position(buy)

    assert len(portfolio.positions) == 1
    pos_key = list(portfolio.positions.keys())[0]
    assert portfolio.positions[pos_key].size == 20

    # Sell ALL 50 shares at $0.50 = $25 notional (not $20 — different price means different notional)
    sell = _make_trade(strategy="mm", price=0.50, size=25, side="sell")
    sell.market_id = "token_123"
    pnl = await portfolio.close_position(sell)

    assert pnl > 0  # bought at 0.40, sold at 0.50
    assert len(portfolio.positions) == 0  # fully closed


@pytest.mark.asyncio
async def test_portfolio_netting_partial_close(portfolio):
    """Selling less than position size should reduce, not close entirely."""
    buy = _make_trade(strategy="mm", price=0.40, size=40, side="buy")
    buy.market_id = "token_123"
    await portfolio.open_position(buy)

    # Sell $15 @ $0.50 = 30 shares closed → remaining = 70 shares * $0.40 = $28
    sell = _make_trade(strategy="mm", price=0.50, size=15, side="sell")
    sell.market_id = "token_123"
    pnl = await portfolio.close_position(sell)

    assert pnl > 0
    assert len(portfolio.positions) == 1
    remaining = list(portfolio.positions.values())[0]
    assert remaining.size == pytest.approx(28.0, abs=0.1)  # 70 shares * $0.40


# --- Reserved cash / available_cash ---

def test_available_cash_default(portfolio):
    """available_cash equals cash when no orders are pending."""
    assert portfolio.available_cash == portfolio.cash
    assert portfolio.reserved_cash == 0.0


def test_available_cash_with_reserved(portfolio):
    """available_cash subtracts reserved_cash."""
    portfolio.reserved_cash = 36.0  # simulate 10 pending BUY orders × $3.60
    assert portfolio.available_cash == pytest.approx(500 - 36, abs=0.01)


def test_available_cash_floor_zero(portfolio):
    """available_cash never goes negative."""
    portfolio.reserved_cash = 9999.0
    assert portfolio.available_cash == 0.0


@pytest.mark.asyncio
async def test_check_can_trade_respects_reserved_cash(rm, portfolio):
    """Risk check should block trades when available_cash is insufficient."""
    portfolio.reserved_cash = 460.0  # only $40 available
    ok, reason = await rm.check_can_trade("test", 50)
    assert not ok
    assert "cash" in reason.lower() or "insufficient" in reason.lower()


@pytest.mark.asyncio
async def test_check_can_trade_passes_with_enough_available(rm, portfolio):
    """Risk check should pass when available_cash covers the trade."""
    portfolio.reserved_cash = 100.0  # $400 available
    ok, reason = await rm.check_can_trade("test", 50)
    assert ok
