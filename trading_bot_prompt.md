# Trading Bot -- Polymarket + Interactive Brokers

## Estado Actual (Phase 6)

| Componente | Estado | Archivo |
|---|---|---|
| Core engine (async event loop) | Active | `core/engine.py` |
| Risk manager (dynamic sizing + degradation) | Active | `core/risk_manager.py` |
| Portfolio tracker | Active | `core/portfolio.py` |
| Execution router (multi-platform) | Active | `core/execution_router.py` |
| Base strategy + Signal validation | Active | `strategies/base_strategy.py` |
| Polymarket connector (WS + CLOB signing) | Active | `connectors/polymarket_client.py` |
| Market maker Avellaneda-Stoikov | **Disabled** | `strategies/poly_market_maker.py` |
| News Edge (LLM directional bets) | **Shadow mode** | `strategies/poly_news_edge.py` |
| LLM client (Claude Haiku) | Active | `connectors/llm_client.py` |
| News scraper (Google RSS) | Active | `connectors/news_scraper.py` |
| IB connector (ib_async) | Conditional | `connectors/ib_client.py` |
| Pairs trading (cointegration) | Conditional | `strategies/ib_pairs.py` |
| Telegram bot (commands + alerts) | Active | `connectors/telegram_bot.py` |
| SQLite WAL database | Active | `data/db.py` |
| Config + secrets management | Active | `config/settings.yaml`, `utils/helpers.py` |
| Logger (rotating file + UTF-8) | Active | `utils/logger.py` |
| Tests (69 passing) | Active | `tests/` |

---

## Estrategias

### 1. Market Making (Avellaneda-Stoikov) -- DISABLED
- Removed from `main.py`, config `enabled: false`
- Lost $7.47 in 6 days (-2.2%) due to structural 2:1 buy:sell ratio at $340 capital
- Code preserved in `strategies/poly_market_maker.py` for future reactivation

### 2. News Edge (Claude Haiku) -- SHADOW MODE
- Shadow mode: logs signals, no real orders
- Pipeline: Google News RSS -> market matching -> Claude Haiku probability estimation
- Filters: price 0.08-0.92, max_end_date_days: 30, min_confidence: 0.50, sports keywords
- BUY only for entry, SELL only for closing positions
- Strategy cap: 10% of equity (~$33)
- TP/SL/timeout: +20% / -15% / 48h
- Validating 5-7 days before going live

### 3. IB Pairs Trading -- CONDITIONAL
- Only activates if `ib.enabled: true` in config (default off)
- Requires IB Gateway/TWS running

---

## Infraestructura

- **VPS:** DigitalOcean $6/mo, Ubuntu 24.04, Bangalore (US geoblock bypass)
- **Deploy:** `sudo bash /opt/tradingbot/deploy/update.sh`
- **Service:** systemd `tradingbot` with auto-restart + hardening
- **Capital:** ~$332 USDC on Polymarket (down from $340 deposit)

---

## Arquitectura

```
trading-bot/
+-- config/
|   +-- settings.yaml              # Centralized config (MM disabled, NE shadow)
|   +-- secrets.env                # API keys (gitignored)
+-- core/
|   +-- engine.py                  # Event loop, signal dispatch, shadow mode, reconciliation
|   +-- execution_router.py        # Routes signals to connectors + balance pre-check
|   +-- risk_manager.py            # Drawdown, exposure cap, cooldown, kill switch
|   +-- portfolio.py               # Positions, PnL, equity, serialization
+-- strategies/
|   +-- base_strategy.py           # ABC + Signal dataclass
|   +-- poly_market_maker.py       # A-S MM (DISABLED)
|   +-- poly_news_edge.py          # News Edge LLM (SHADOW)
|   +-- ib_pairs.py                # Cointegration pairs (CONDITIONAL)
+-- connectors/
|   +-- polymarket_client.py       # WS market+user, REST orders, CLOB signing
|   +-- llm_client.py              # Claude Haiku probability estimation
|   +-- news_scraper.py            # Google News RSS scraper
|   +-- ib_client.py               # ib_async wrapper
|   +-- telegram_bot.py            # /status /pnl /trades /stop /start /reset /risk
+-- data/
|   +-- db.py                      # SQLite WAL + async write queue + state persistence
+-- utils/
|   +-- logger.py                  # Rotating file + UTF-8 console
|   +-- helpers.py                 # Config loader, env var resolution
+-- tests/                         # 69 tests passing
+-- main.py                        # Entry point (NE only)
```

---

## Fases

### Phase 1 -- MVP Polymarket -- COMPLETED
- Core engine, risk, portfolio, polymarket connector, MM, telegram, db, tests

### Phase 2 -- IB Pairs Paper -- COMPLETED
- IB connector, pairs trader, dynamic cointegration scanner

### Phase 3 -- Live Deployment -- COMPLETED
- VPS deployment, systemd service, live trading with MM

### Phase 4 -- News Edge Development -- COMPLETED
- LLM client, news scraper, NE strategy, shadow mode, filters

### Phase 5 -- MM Tuning + NE Filters -- COMPLETED
- Inventory control, BUY gates, NE confidence/expiry filters, balance pre-check

### Phase 6 -- MM Disabled, NE Validation -- CURRENT
- MM disabled (structural losses at $340)
- NE shadow-only: validating signal quality 5-7 days
- Next: if NE hit_rate > 52% and positive expectancy -> go live

### Phase 7 -- Scale (future)
- Capital $1,000+, re-evaluate MM viability, IB pairs live
