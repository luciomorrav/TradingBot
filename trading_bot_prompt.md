# Trading Bot — Polymarket + Interactive Brokers + Kalshi

## Estado de Implementación

| Componente | Estado | Archivo |
|---|---|---|
| Core engine (async event loop) | ✅ Implementado | `core/engine.py` |
| Risk manager (dynamic sizing + degradation) | ✅ Implementado | `core/risk_manager.py` |
| Portfolio tracker | ✅ Implementado | `core/portfolio.py` |
| Base strategy + Signal validation | ✅ Implementado | `strategies/base_strategy.py` |
| Polymarket connector (WebSocket-first) | ✅ Implementado | `connectors/polymarket_client.py` |
| Market maker Avellaneda-Stoikov | ✅ Implementado | `strategies/poly_market_maker.py` |
| IB connector (ib_async + auto-reconnect) | ✅ Implementado | `connectors/ib_client.py` |
| Pairs trading (cointegración dinámica) | ✅ Implementado | `strategies/ib_pairs.py` |
| Telegram bot (comandos + notificaciones) | ✅ Implementado | `connectors/telegram_bot.py` |
| SQLite WAL database | ✅ Implementado | `data/db.py` |
| Config + secrets management | ✅ Implementado | `config/settings.yaml`, `utils/helpers.py` |
| Logger (rotating file + console) | ✅ Implementado | `utils/logger.py` |
| Entry point + wiring | ✅ Implementado | `main.py` |
| Security audit | ✅ Auditado (1 HIGH + 5 MED + 2 LOW corregidos) |
| Execution router (multi-platform) | ✅ Implementado | `core/execution_router.py` |
| Polymarket live orders (py-clob-client) | ✅ Implementado | `connectors/polymarket_client.py` |
| Tests unitarios (30 tests) | ✅ Implementado | `tests/test_risk.py`, `tests/test_strategies.py` |
| News edge (pipeline 2 etapas) | ⏳ Fase 2 | `strategies/poly_news_edge.py` |
| Probability model LLM | ⏳ Fase 2 | `strategies/poly_probability_model.py` |
| LLM client (Claude API) | ⏳ Fase 2 | `connectors/llm_client.py` |
| News scraper (RSS + APIs) | ⏳ Fase 2 | `connectors/news_scraper.py` |
| Cross-platform arb Poly/Kalshi | ⏳ Fase 4 | `strategies/cross_platform_arb.py` |
| Kalshi connector | ⏳ Fase 4 | `connectors/kalshi_client.py` |

---

## Contexto

Inversor con cuenta en Interactive Brokers y experiencia en Polymarket. Bot modular en Python que opera de forma autónoma con supervisión vía Telegram. Stack: Python 3.12+, asyncio, WebSocket-first.

**Capital inicial: $500 USD**
- $400 → Polymarket market making (live)
- $100 → Reserva para gas/fees en Polygon
- $0 → IB pairs trading (paper only hasta validar)

---

## Arquitectura Actual

```
trading-bot/
├── config/
│   ├── settings.yaml              # Config centralizada
│   └── secrets.env                # API keys (gitignored)
├── core/
│   ├── engine.py                  # Event-driven async loop, strategy orchestration
│   ├── risk_manager.py            # Dynamic sizing, kill switch, execution degradation
│   └── portfolio.py               # Posiciones, PnL, exposure, fees, LLM cost tracking
├── strategies/
│   ├── base_strategy.py           # ABC + Signal (validado: direction, price, size, confidence)
│   ├── poly_market_maker.py       # Avellaneda-Stoikov + informed flow + fee awareness
│   ├── ib_pairs.py                # Cointegración dinámica + z-score mean reversion
│   ├── poly_news_edge.py          # [FASE 2] Pipeline 2 etapas: rules → Claude haiku
│   ├── poly_probability_model.py  # [FASE 2] Modelo LLM direccional
│   └── cross_platform_arb.py      # [FASE 4] Polymarket vs Kalshi
├── connectors/
│   ├── polymarket_client.py       # WebSocket market data + REST orders
│   ├── ib_client.py               # ib_async wrapper + auto-reconnect + streaming
│   ├── telegram_bot.py            # /status /trades /stop /start /risk /pnl
│   ├── llm_client.py              # [FASE 2] Claude API wrapper
│   ├── kalshi_client.py           # [FASE 4] Kalshi API
│   └── news_scraper.py            # [FASE 2] RSS + APIs
├── data/
│   └── db.py                      # SQLite WAL + async write queue + índices
├── utils/
│   ├── logger.py                  # Rotating file (100MB x7) + console
│   └── helpers.py                 # Config loader, env var resolution, validation
├── tests/                         # [PENDIENTE]
├── main.py                        # Entry point, wires all components
├── requirements.txt
└── .gitignore
```

---

## Módulo 1: Core Engine

### engine.py — Implementado
- Event-driven async loop (asyncio)
- Strategy registration con intervalo configurable por strategy
- Ciclo: `run_cycle()` → `evaluate()` → risk check → `execute` → log → notify
- Paper mode: simula ejecución instantánea al precio de señal
- Live mode: delega a executor callback (connector)
- Background loops: heartbeat (30min), daily risk reset (midnight UTC)
- Graceful shutdown: SIGINT/SIGTERM (Unix), cancela todas las tasks
- Notificaciones Telegram en cada trade y al arrancar/parar

### risk_manager.py — Implementado
- **Dynamic position sizing** basado en volatilidad (no porcentaje fijo)
- Max loss por estrategia individual + global
- Max exposición total configurable (60% default)
- **Kill switch**: drawdown > 5% → detener todo + notificar Telegram
- Cooldown: 3 trades perdedores consecutivos → pausar 30 min
- **Execution degradation detection**:
  - Fill rate < 80% → reducir tamaño
  - Slippage promedio > 1% → pausar estrategia
  - Latencia > 5s → alertar y reducir
- Rolling window de métricas de ejecución por estrategia
- Config via `RiskConfig.from_dict()` (settings.yaml)

### portfolio.py — Implementado
- Tracking en tiempo real de posiciones (Polymarket + IB)
- PnL realizado y no realizado
- Exposure neto y por posición
- Tracking de fees + costos Claude API
- `deque(maxlen=10000)` para closed trades (bounded memory)
- Async locks en `update_price()` y `add_llm_cost()`

---

## Módulo 2: Polymarket Strategies

### poly_market_maker.py — Implementado (Avellaneda-Stoikov)
Lógica:
1. **Selección de mercados**: volumen $500-5000/día, max 5 mercados simultáneos
2. **Modelo Avellaneda-Stoikov**:
   - Reservation price = mid - q * γ * σ² * T
   - Optimal spread = γ * σ² * T + (2/γ) * ln(1 + γ/κ)
   - Parámetros: γ=0.1 (risk aversion), κ=1.5 (order arrival intensity)
3. **Inventory-adjusted quotes**: sesgo hacia reducir inventario
4. **Informed flow detection**: top-level size > $500 → ampliar spread
5. **Fee awareness**: fees dinámicos por mercado (0% global markets, hasta 3% crypto)
6. **Cancel/repost**: si mercado mueve > 2c
7. Max inventory por mercado: $200

### poly_news_edge.py — FASE 2 (no implementado)
Pipeline 2 etapas:
1. Filtro barato (rules/keywords) sobre RSS/APIs
2. Claude haiku solo si pasa filtro → clasificar relevancia + estimar dirección
3. Ejecutar si confidence > 70% y mercado no ajustó
4. Latencia objetivo: < 30s
5. Max costo Claude: $0.50/día

### poly_probability_model.py — FASE 2 (no implementado)
- Claude haiku estima probabilidad real del evento
- Compara vs precio de mercado
- Posición si diferencia > 5%
- Re-evaluar cada 4-6 horas
- Paper hasta accuracy > 60%

### cross_platform_arb.py — FASE 4 (no implementado)
- Polymarket vs Kalshi con normalización de contratos
- Verificar reglas de resolución, expiración, tamaños mínimos
- Solo pares 100% verificados
- Solo con capital > $5,000

---

## Módulo 3: IB Pairs Trading

### ib_pairs.py — Implementado (Scanner Dinámico de Cointegración)

**Cambio vs spec original**: se eliminaron los pares hardcodeados de ADRs argentinos. El scanner ahora encuentra los mejores pares dinámicamente de un universo amplio.

**Universo configurable** (~30 símbolos, ~435 combinaciones):
- Sector ETFs: XLE, XOP, XLF, KBE, GDX, GDXJ, XLK, VGT, XLV, XBI
- Large-cap: KO, PEP, V, MA, JPM, BAC, HD, LOW, MSFT, AAPL
- Commodities: GLD, GDX, USO, SLV, PAAS
- Index: SPY, IVV, QQQ, QQQM, IWM, VTWO
- Dual-class: GOOG/GOOGL

**Lógica del scanner**:
1. Carga data histórica (120 días) de todo el universo al arrancar
2. Testea cointegración Engle-Granger (ADF sobre residuos OLS)
3. Calcula half-life de mean reversion (AR(1) sobre residuos)
4. Filtra: p-value < 0.05, half-life entre 5 y 60 días
5. Rankea por p-value (menor = mejor)
6. Opera top 5 pares simultáneos
7. Re-escanea cada 24 horas, rota pares que pierden cointegración
8. Preserva pares con posición abierta al re-escanear

**Señales de trading**:
- Entry: z-score > 2.0 (short spread) o < -2.0 (long spread)
- Exit: z-score cruza ±0.5
- Stop: z-score > 3.5 o < -3.5
- Size: $50/leg (conservador para paper)

### ib_client.py — Implementado
- Conexión via `ib_async` (NO ib_insync — archivado marzo 2024)
- Auto-reconnect con exponential backoff (5s → 120s max)
- Historical data: `reqHistoricalData` async con timeout
- Real-time: `reqMktData` + `pendingTickersEvent`
- Orders: `MarketOrder` / `LimitOrder` con espera de fill
- `get_positions()`, `get_account_value()`
- Re-subscribe automático tras reconexión
- Puertos: 4002 (paper), 4001 (live)

---

## Módulo 4: Telegram Bot — Implementado

Comandos:
- `/status` → Equity, cash, exposure, PnL, drawdown, posiciones, fees
- `/trades` → Últimos 10 trades
- `/stop` → Kill switch manual
- `/start` → Reanudar trading
- `/risk` → Estado de risk, degradation metrics, paused strategies
- `/pnl` → PnL realizado + no realizado + fees + LLM cost

Notificaciones automáticas:
- Cada trade ejecutado (dirección, size, symbol, precio, PnL)
- Bot start/stop
- Heartbeat cada 30 min (equity, PnL, posiciones)

**Pendiente**: `/audit`, `/compare`, `/set`, resumen diario automático

---

## Módulo 5: Configuración

### settings.yaml — Actualizado
```yaml
general:
  mode: paper
  capital_usd: 500
  log_level: INFO

polymarket:
  api_key: ${POLY_API_KEY}
  chain_id: 137
  min_profit_threshold: 0.015
  ws_url: wss://ws-subscriptions-clob.polymarket.com/ws/market
  max_position_per_market: 100
  market_maker:
    target_spread: 0.05
    max_inventory: 200
    repost_threshold: 0.02

ib:
  host: 127.0.0.1
  port: 4002
  client_id: 1
  universe:
    sector_etfs: [XLE, XOP, XLF, KBE, GDX, GDXJ, XLK, VGT, XLV, XBI]
    large_cap: [KO, PEP, V, MA, JPM, BAC, HD, LOW, MSFT, AAPL]
    commodities: [GLD, GDX, USO, SLV, PAAS]
    index: [SPY, IVV, QQQ, QQQM, IWM, VTWO]
    dual_class: [[GOOG, GOOGL]]
  scanner:
    lookback_days: 120
    rescan_hours: 24
    max_active_pairs: 5
    min_half_life: 5
    max_half_life: 60
  zscore_entry: 2.0
  zscore_exit: 0.5
  zscore_stop: 3.5
  cointegration_pvalue: 0.05

kalshi:  # Fase 4
  api_key: ${KALSHI_API_KEY}
  demo_mode: true

llm:
  provider: anthropic
  model: claude-haiku-4-5-20251001
  fallback_model: claude-sonnet-4-6
  validate_model_on_startup: true
  api_key: ${ANTHROPIC_API_KEY}
  max_cost_per_day: 0.50
  news_edge_confidence_threshold: 0.70
  probability_model_diff_threshold: 0.05

risk:
  max_daily_drawdown_pct: 5.0
  max_position_pct: 20.0
  max_total_exposure_pct: 60.0
  consecutive_loss_cooldown: 3
  cooldown_minutes: 30
  execution_degradation:
    min_fill_rate: 0.80
    max_slippage_pct: 1.0
    max_latency_seconds: 5

telegram:
  bot_token: ${TELEGRAM_BOT_TOKEN}
  chat_id: ${TELEGRAM_CHAT_ID}
  daily_summary_hour_utc: 22
```

### Env vars requeridos (secrets.env)
- `POLY_API_KEY` — Polymarket CLOB API key
- `TELEGRAM_BOT_TOKEN` — Bot de @BotFather
- `TELEGRAM_CHAT_ID` — Chat ID para notificaciones
- `ANTHROPIC_API_KEY` — (Fase 2)
- `KALSHI_API_KEY` — (Fase 4)

---

## Módulo 6: Deployment

**Fase 1 (actual — Polymarket live + IB paper):**
- Puede correr local o VPS barato ($5/mes)
- Para IB paper trading: **requiere TWS o IB Gateway corriendo** (la API no funciona sin uno de los dos)
  - TWS (interfaz gráfica): más fácil para dev local
  - IB Gateway (headless): preferible para VPS/servidor
  - Puerto 4002 (paper) — configurado en settings.yaml
- Si IB no está disponible, el bot arranca sin pairs trading (se deshabilita automáticamente)

**Fase 3+ (IB live):**
- IB Gateway + Xvfb + IBC en VPS (o Docker: `gnzsnz/ib-gateway-docker`)
- Puerto 4001 (live) — cambiar en settings.yaml
- systemd o docker-compose

**Seguridad (auditada):**
- secrets.env con permisos 600
- aiohttp >= 3.13.3 (CVE fix)
- Signal validation (direction, price, size, confidence)
- Env vars fail-fast al arrancar
- Background loops con try/except/CancelledError
- closed_trades bounded (deque maxlen=10000)
- update_price con async lock
- Log level allowlist
- Config validation (capital > 0, mode valid, risk params positive)

---

## Dependencias

```
ib_async>=1.0.0            # IB (sucesor de ib_insync archivado)
py-clob-client>=0.34       # Polymarket CLOB
anthropic>=0.40            # Claude API (Fase 2)
python-telegram-bot==20.7
aiohttp>=3.13.3            # Pinned por CVE
pyyaml>=6.0
python-dotenv>=1.0
numpy>=1.26
pandas>=2.1
scipy>=1.12
statsmodels>=0.14          # Engle-Granger cointegration
aiosqlite>=0.19
feedparser>=6.0            # (Fase 2)
```

---

## Fases

### Fase 1A — MVP Polymarket ✅ COMPLETADO
1. ✅ core/engine.py + risk_manager.py + portfolio.py
2. ✅ connectors/polymarket_client.py (WebSocket-first)
3. ✅ strategies/poly_market_maker.py (Avellaneda-Stoikov)
4. ✅ connectors/telegram_bot.py
5. ✅ data/db.py (SQLite WAL)
6. ✅ Security audit (1 HIGH + 5 MED + 2 LOW fixed)
7. ⏳ Deploy local → paper trade 2+ semanas → live con $400

### Fase 1B — IB Pairs Paper ✅ COMPLETADO
8. ✅ connectors/ib_client.py (ib_async + auto-reconnect)
9. ✅ strategies/ib_pairs.py (scanner dinámico de cointegración)
10. Paper indefinido hasta validar

### Fase 2 — LLM (cuando fase 1 sea rentable)
11. connectors/llm_client.py (Claude API wrapper)
12. connectors/news_scraper.py
13. strategies/poly_news_edge.py (pipeline 2 etapas)
14. strategies/poly_probability_model.py
15. Paper trade LLM strategies 1+ mes

### Fase 3 — Escalar (si 2 meses rentables)
16. Meter más capital ($1,000-2,000)
17. Activar IB pairs live
18. Evaluar news edge live
19. Setup IB Gateway en VPS

### Fase 4 — Cross-Platform Arb (solo si capital > $5,000)
20. connectors/kalshi_client.py
21. strategies/cross_platform_arb.py (normalización de contratos)

---

## Qué Falta para Paper Trading Real

Para poder ejecutar el bot en paper mode:

1. **Secrets configurados**: crear `config/secrets.env` con `POLY_PRIVATE_KEY`, `POLY_API_KEY`/`SECRET`/`PASSPHRASE`, y `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`
2. ~~**Tests unitarios**~~: ✅ 30 tests implementados (risk, sizing, cointegración, A-S model, inventory tracking)
3. **Paper execution de Polymarket**: simula fills instantáneos — considerar simular latencia y slippage realistas
4. **IB Gateway/TWS corriendo**: necesario para pairs trading (se deshabilita automáticamente si IB no está disponible)
5. **Validar con data real**: el market maker necesita mercados activos en Polymarket; el pairs trader necesita data histórica de IB

## Qué Falta para Live Trading

Requisitos adicionales antes de operar con dinero real:

1. ~~**ExecutionRouter**~~: ✅ Implementado — routea signals a Polymarket o IB según plataforma
2. ~~**Polymarket signed orders**~~: ✅ Integrado py-clob-client con EIP-712 signing (L2 auth)
3. ~~**Quote management real**~~: ✅ Inventory from fills, order TTL, cancel/repost lifecycle
4. **Allowances en Polygon**: aprobar USDC y Conditional Tokens para los contratos de exchange
5. **Paper trade 2-4 semanas**: métricas duras antes de ir live
6. **Ejecución atómica IB pairs**: fail-safe si una leg falla (hedge inmediato de la otra)

---

## KPIs Operativos

| KPI | Target | Nota |
|---|---|---|
| Sharpe ratio | > 1.5 | Rolling 30 días |
| Max drawdown | < 5% | $25 con $500 capital |
| Fill rate | > 90% | Market maker |
| Slippage promedio | < 0.5% | Vs precio esperado |
| PnL neto de fees | Positivo | Rolling 30 días |
| Costo Claude API | < $20/mes | Fase 2 |
| Objetivo mínimo | No perder dinero | Primeros 2 meses |

### Criterios de Escalado

| Condición | Acción |
|---|---|
| 2 meses rentables, Sharpe > 1.0 | Escalar a $1,000-2,000 |
| 4 meses rentables, Sharpe > 1.5 | Escalar a $5,000+, activar IB live |
| Drawdown > 10% ($50+ perdidos) | Pausar, analizar, ajustar parámetros |
| 2 meses con pérdida | Volver a paper, no meter más capital |
| Estrategia pierde 3 meses seguidos | Desactivar estrategia |

---

## Sistema de Auditorías

### Métricas trackeadas por estrategia
- PnL diario/semanal/mensual (realizado + no realizado)
- Win rate y profit factor
- Sharpe ratio rolling (30 días)
- Max drawdown del período
- Trades y frecuencia
- Costo Claude API
- Slippage promedio
- Fill rate

### Auditoría Semanal (domingos)
1. Performance por estrategia
2. Risk check: drawdown, exposición
3. Ejecución: slippage, fills
4. Costos: fees Poly + comisiones IB + Claude API
5. Errores: reconexiones, timeouts, fallos

### Auditoría Mensual
- Go/Kill por estrategia (pierde 2 meses → pausar, Sharpe < 0.5 → revisar)
- Rebalanceo de capital entre estrategias
- Calibración de parámetros (z-score, spread, confidence thresholds)

### Auditoría Trimestral
- Cambios de mercado (fees, regulación, competencia)
- Nuevas oportunidades (plataformas, instrumentos, datos)
- Escalar o pivotar
