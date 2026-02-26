# Trading Bot — Polymarket + Interactive Brokers + Kalshi

## Estado de Implementacion

| Componente | Estado | Archivo |
|---|---|---|
| Core engine (async event loop) | Done | `core/engine.py` |
| Risk manager (dynamic sizing + degradation) | Done | `core/risk_manager.py` |
| Portfolio tracker | Done | `core/portfolio.py` |
| Base strategy + Signal validation | Done | `strategies/base_strategy.py` |
| Polymarket connector (WS + CLOB signing) | Done | `connectors/polymarket_client.py` |
| Market maker Avellaneda-Stoikov | Done | `strategies/poly_market_maker.py` |
| IB connector (ib_async + auto-reconnect) | Done | `connectors/ib_client.py` |
| Pairs trading (cointegration dinamica) | Done | `strategies/ib_pairs.py` |
| Telegram bot (comandos + notificaciones) | Done | `connectors/telegram_bot.py` |
| SQLite WAL database | Done | `data/db.py` |
| Config + secrets management | Done | `config/settings.yaml`, `utils/helpers.py` |
| Logger (rotating file + UTF-8 console) | Done | `utils/logger.py` |
| Entry point + wiring | Done | `main.py` |
| Execution router (multi-platform) | Done | `core/execution_router.py` |
| Polymarket live orders (py-clob-client) | Done | `connectors/polymarket_client.py` |
| Tests unitarios (30 tests) | Done | `tests/test_risk.py`, `tests/test_strategies.py` |
| Paper mode runtime verificado | Done | Ran 90s+, signals generated, fills tracked |
| Security audit | Done | 1 HIGH + 5 MED + 2 LOW corregidos |
| News edge (pipeline 2 etapas) | Fase 2 | `strategies/poly_news_edge.py` |
| Probability model LLM | Fase 2 | `strategies/poly_probability_model.py` |
| LLM client (Claude API) | Fase 2 | `connectors/llm_client.py` |
| News scraper (RSS + APIs) | Fase 2 | `connectors/news_scraper.py` |
| Cross-platform arb Poly/Kalshi | Fase 4 | `strategies/cross_platform_arb.py` |
| Kalshi connector | Fase 4 | `connectors/kalshi_client.py` |

---

## Cuentas Configuradas

| Plataforma | Estado | Detalle |
|---|---|---|
| Polymarket | Listo | Wallet magic, private key + CLOB API L2 auth, $339.87 USDC |
| Telegram | Listo | @Morris_Trading_bot, token + chat_id configurados |
| Interactive Brokers | Pendiente | Necesita IB Gateway/TWS para paper trading |
| Anthropic (Claude API) | Fase 2 | Solo cuando se implementen estrategias LLM |
| Kalshi | Fase 4 | Solo con capital > $5,000 |

---

## Contexto

Bot modular en Python que opera de forma autonoma con supervision via Telegram. Stack: Python 3.14, asyncio, WebSocket-first.

**Capital actual: $340 USD (Polymarket)**
- $340 en Polymarket (USDC en Polygon)
- IB: paper only (no requiere capital)

---

## Arquitectura

```
trading-bot/
+-- config/
|   +-- settings.yaml              # Config centralizada
|   +-- secrets.env                # API keys (gitignored)
|   +-- secrets.env.example        # Template
+-- core/
|   +-- engine.py                  # Event-driven async loop, strategy orchestration
|   +-- execution_router.py        # Routes signals to Polymarket or IB
|   +-- risk_manager.py            # Dynamic sizing, kill switch, execution degradation
|   +-- portfolio.py               # Posiciones, PnL, exposure, fees
+-- strategies/
|   +-- base_strategy.py           # ABC + Signal (validado: direction, price, size, confidence)
|   +-- poly_market_maker.py       # Avellaneda-Stoikov + informed flow + fee awareness
|   +-- ib_pairs.py                # Cointegration dinamica + z-score mean reversion
+-- connectors/
|   +-- polymarket_client.py       # WebSocket market data + REST CLOB orders (signed)
|   +-- ib_client.py               # ib_async wrapper + auto-reconnect + streaming
|   +-- telegram_bot.py            # /status /trades /stop /start /risk /pnl
+-- data/
|   +-- db.py                      # SQLite WAL + async write queue
+-- utils/
|   +-- logger.py                  # Rotating file (100MB x7) + UTF-8 console
|   +-- helpers.py                 # Config loader, env var resolution, validation
+-- tests/
|   +-- test_risk.py               # 15 tests: risk manager
|   +-- test_strategies.py         # 15 tests: signals, cointegration, A-S model
+-- main.py                        # Entry point
+-- .gitignore
```

---

## Que se verifico en Paper Mode

Bot corrido exitosamente en paper mode:
1. Database SQLite WAL conecta OK
2. Telegram bot arranca, manda notificacion de inicio, polling activo
3. Polymarket CLOB client inicializa con L2 auth (live orders enabled)
4. Gamma API fetcha 31,000+ mercados activos
5. Market maker selecciona 8 mercados, suscribe 16 tokens via WebSocket
6. WebSocket conecta y recibe orderbooks en tiempo real
7. Estrategia genera senales y ejecuta paper trades (A-S model funcional)
8. Fills se trackean en inventory del market maker
9. IB se deshabilita gracefully si no hay Gateway corriendo
10. 30/30 tests pasan

**Issues arreglados durante runtime:**
- python-telegram-bot 20.7 incompatible con httpx 0.28 -> upgrade a 22.6
- YAML parsea `0x...` como hex int -> conversion explicita a string
- Gamma API devuelve JSON strings -> json.loads()
- Emojis crashean Windows console (cp1252) -> UTF-8 wrapper
- target_spread 0.05 demasiado alto para mercados reales -> 0.02

---

## Modulo 1: Core Engine

### engine.py
- Event-driven async loop (asyncio)
- Strategy registration con intervalo configurable
- Paper mode: simula fill instantaneo al precio de senal
- Live mode: delega a ExecutionRouter (polymarket/ib)
- Background: heartbeat (30min), daily risk reset (midnight UTC)
- Graceful shutdown: cancela tasks, notifica Telegram
- `_notify_strategy_fill()`: conecta fills con inventory del market maker

### risk_manager.py
- **Dynamic sizing** basado en volatilidad
- Max exposicion total: 60% de capital
- **Kill switch**: drawdown > 5% -> detener todo + Telegram
- Cooldown: 3 losses consecutivos -> pausar 30 min
- **Execution degradation**: fill rate < 80%, slippage > 1%, latencia > 5s
- Rolling window metricas por estrategia

### portfolio.py
- Tracking real-time posiciones (Polymarket + IB)
- PnL realizado y no realizado
- Exposure neto y por posicion/estrategia/plataforma
- Fees + LLM cost tracking
- Async locks, bounded deque (10000 closed trades)

---

## Modulo 2: Polymarket Market Maker

### poly_market_maker.py (Avellaneda-Stoikov)
1. **Seleccion de mercados**: volumen $200-10000/dia, max 8 mercados
2. **Modelo A-S**: reservation_price = mid - q*gamma*sigma^2*T, optimal_spread = gamma*sigma^2*T + (2/gamma)*ln(1+gamma/kappa)
3. **Inventory-adjusted quotes**: sesgo hacia reducir inventario
4. **Informed flow detection**: top-level size > $500 -> skip
5. **Fee awareness**: fees dinamicos por mercado
6. **Order lifecycle**: LiveOrder con TTL, cancel stale, repost si market mueve > 2c
7. **Fill tracking**: on_fill() actualiza inventory de fills reales, track_order() para paper

### polymarket_client.py
- WebSocket para market data (no rate limits)
- REST para ordenes (60/min limit)
- py-clob-client para signed orders (L2 auth: HMAC-SHA256)
- Paper mode: simula fills instantaneos
- Live mode: create_order + post_order (EIP-712 signing)
- cancel_order, cancel_all_orders
- Gamma API para market discovery (31,000+ mercados)

---

## Modulo 3: IB Pairs Trading

### ib_pairs.py (Scanner Dinamico)
- Universo: ~30 simbolos, ~435 combinaciones
- Engle-Granger cointegration (ADF sobre residuos OLS)
- Half-life AR(1): filtro 5-60 dias
- Top 5 pares por p-value
- Re-scan cada 24h, preserva posiciones abiertas
- Senales: entry z>2.0, exit z<0.5, stop z>3.5

### ib_client.py
- ib_async (sucesor de ib_insync)
- Auto-reconnect exponential backoff (5s->120s)
- Historical data, streaming, orders
- Se deshabilita automaticamente si no hay Gateway

---

## Modulo 4: Telegram Bot

Comandos: /status /trades /stop /start /risk /pnl
Notificaciones: cada trade, start/stop, heartbeat 30min

---

## Secrets (config/secrets.env)

```
POLY_PRIVATE_KEY=0x...          # Wallet Polygon (magic wallet)
POLY_API_KEY=...                # CLOB API (derivado de PK)
POLY_API_SECRET=...             # CLOB API
POLY_API_PASSPHRASE=...         # CLOB API
TELEGRAM_BOT_TOKEN=...          # @Morris_Trading_bot
TELEGRAM_CHAT_ID=...            # Chat ID del user
ANTHROPIC_API_KEY=...           # (Fase 2)
KALSHI_API_KEY=...              # (Fase 4)
```

---

## Dependencias Principales

```
ib_async>=1.0.0
py-clob-client>=0.34
python-telegram-bot>=22.6
aiohttp>=3.13.3
pyyaml, python-dotenv, numpy, statsmodels, aiosqlite
```

---

## Fases

### Fase 1A — MVP Polymarket -> COMPLETADO
- Core engine, risk, portfolio, polymarket connector, market maker, telegram, db
- Security audit, execution router, signed orders, tests

### Fase 1B — IB Pairs Paper -> COMPLETADO
- IB connector, pairs trader, dynamic cointegration scanner

### Fase 1C — Paper Trading Validacion -> EN CURSO
- Paper mode verificado, bot corre exitosamente
- Pendiente: correr 2-4 semanas continuo, recopilar metricas
- Deployment en VPS para uptime 24/7

### Fase 2 — LLM Strategies (cuando fase 1 sea rentable)
- llm_client.py (Claude haiku), news_scraper.py
- poly_news_edge.py (pipeline 2 etapas: rules -> haiku)
- poly_probability_model.py

### Fase 3 — Escalar (si 2 meses rentables)
- Capital $1,000-2,000, IB pairs live, IB Gateway en VPS

### Fase 4 — Cross-Platform Arb (capital > $5,000)
- Kalshi connector + arb strategy

---

## KPIs Operativos

| KPI | Target |
|---|---|
| Sharpe ratio | > 1.5 (rolling 30d) |
| Max drawdown | < 5% |
| Fill rate | > 90% |
| Slippage promedio | < 0.5% |
| PnL neto de fees | Positivo (rolling 30d) |
| Objetivo minimo | No perder dinero primeros 2 meses |

### Escalado
- 2 meses rentables, Sharpe > 1.0 -> $1,000-2,000
- 4 meses rentables, Sharpe > 1.5 -> $5,000+, IB live
- Drawdown > 10% -> pausar, analizar
- 2 meses con perdida -> volver a paper
