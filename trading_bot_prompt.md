# PROMPT PARA CLAUDE CODE — Trading Bot (Polymarket + Interactive Brokers + Kalshi)

## CONTEXTO DEL PROYECTO

Soy un inversor con cuenta en Interactive Brokers y experiencia operando en Polymarket. Necesito un bot de trading modular en Python que opere de forma autónoma con supervisión vía Telegram. El stack debe ser: Python 3.12+, con dependencias mínimas y código limpio.

El bot usa Claude API (haiku) para análisis de noticias y estimación de probabilidades.

**Capital inicial confirmado: $500 USD**
- $400 → Polymarket market making (live)
- $100 → Reserva para gas/fees en Polygon
- $0 → IB pairs trading (solo paper, sin capital real hasta validar)
- Escalar si las estrategias son rentables (ver criterios de escalado en Módulo 7)

## ARQUITECTURA REQUERIDA

```
trading-bot/
├── config/
│   ├── settings.yaml          # Toda la config centralizada
│   └── secrets.env            # API keys (gitignored)
├── core/
│   ├── engine.py              # Event-driven async loop (WebSocket-first)
│   ├── risk_manager.py        # Dynamic sizing, per-strategy limits, execution degradation
│   └── portfolio.py           # Estado del portfolio en memoria
├── strategies/
│   ├── base_strategy.py       # Clase abstracta Strategy
│   ├── poly_market_maker.py   # Market making Avellaneda-Stoikov en Polymarket
│   ├── poly_news_edge.py      # Pipeline 2 etapas: rules → Claude haiku
│   ├── poly_probability_model.py  # Modelo de probabilidad LLM direccional
│   ├── cross_platform_arb.py  # Arbitraje Polymarket vs Kalshi (fase 4)
│   └── ib_pairs.py            # Pairs trading en IB (ADRs argentinos)
├── connectors/
│   ├── polymarket_client.py   # API Polymarket WebSocket-first (market + user channels)
│   ├── kalshi_client.py       # API Kalshi demo primero (fase 4)
│   ├── ib_client.py           # Interactive Brokers via ib_async
│   ├── llm_client.py          # Wrapper Claude API (haiku)
│   ├── telegram_bot.py        # Notificaciones + comandos
│   └── news_scraper.py        # RSS + APIs de noticias
├── data/
│   ├── market_data.py         # Feeds de precios unificado (WebSocket)
│   └── db.py                  # SQLite WAL + índices + cola async escrituras
├── utils/
│   ├── logger.py              # Logging estructurado
│   └── helpers.py             # Utilidades comunes
├── tests/
│   ├── test_strategies.py
│   └── test_risk.py
├── main.py                    # Entry point
├── requirements.txt
└── README.md
```

## MÓDULO 1: CORE ENGINE (Implementar primero)

### engine.py
- Event-driven async loop (asyncio), WebSocket-first para Polymarket y IB
- Procesos desacoplados: market-data → strategy → execution → risk
- Ciclo: receive_event → evaluate_signals → check_risk → execute → log → notify
- Intervalo configurable por strategy (ej: market maker reactivo a WS, pairs cada 60s)
- Graceful shutdown con signal handling
- Estado persistente en SQLite (último estado del bot, posiciones abiertas)

### risk_manager.py
- Dynamic position sizing basado en volatilidad reciente (no porcentaje fijo)
- Max loss por estrategia individual (no solo global)
- Max exposición total configurable
- Kill switch: si drawdown > X%, detener todo y notificar por Telegram
- Cooldown después de N trades perdedores consecutivos (default: 3)
- Circuit breaker por degradación de ejecución:
  - Fill rate cae por debajo de 80% → reducir tamaño de órdenes
  - Slippage promedio sube > 1% → pausar estrategia
  - Latencia de ejecución > 5s → alertar y reducir actividad
- Correlación entre estrategias: si market making y news edge están en el mismo mercado, sumar exposición

### portfolio.py
- Tracking en tiempo real de posiciones abiertas (Polymarket + IB + Kalshi)
- PnL realizado y no realizado
- Cálculo de exposición neta
- Tracking de fees y costos de Claude API

## MÓDULO 2: POLYMARKET STRATEGIES (Prioridad alta)

### poly_market_maker.py — Market Making (Avellaneda-Stoikov)
Lógica:
1. Identificar mercados nicho con spread > 5 centavos y volumen $500-5000/día (evitar mercados donde compiten bots grandes)
2. Calcular quotes óptimos usando modelo Avellaneda-Stoikov:
   - Mid price del order book
   - Inventory risk (sesgo de quotes según posición actual)
   - Volatilidad estimada del mercado
   - Time to resolution (reducir exposición cerca del vencimiento)
3. Postear limit orders en ambos lados (bid y ask)
4. Detección de informed flow: si entran órdenes grandes (>$500), ampliar spread o salir temporalmente
5. Aprovechar Polymarket Liquidity Rewards (rebates diarios para market makers)
6. Cancelar y repostear si el mercado se mueve > 2 centavos
7. Max inventory por mercado configurable
8. Usar WebSocket (market channel) para reaccionar en tiempo real, no polling
9. **Fees dinámicos**: calcular fees reales por mercado/token antes de cada trade (no asumir fee fijo). Polymarket tiene fees variables: 0% en muchos mercados globales, hasta 3% taker en crypto 15-min markets, fees específicos en sports. Consultar fee schedule via API o config por tipo de mercado.

### poly_news_edge.py — Edge por noticias (Pipeline 2 etapas)
Lógica:
1. **Etapa 1 — Filtro barato (rules/keywords):**
   - Monitorear RSS feeds y APIs de noticias (NewsAPI, Reuters RSS, Twitter/X)
   - Filtrar por keywords relevantes a mercados abiertos de Polymarket
   - Solo las noticias que pasan el filtro van a etapa 2
2. **Etapa 2 — Claude API (haiku):**
   - Clasificar relevancia de la noticia vs mercados abiertos (0-100 score)
   - Estimar impacto direccional (YES/NO) con confidence score
   - Determinar si el mercado ya ajustó o no (comparar precio actual vs hace 60s)
3. Si confidence > 70% y mercado no ajustó → ejecutar
4. Position sizing proporcional a confidence score
5. Latencia objetivo: < 30 segundos desde publicación de noticia hasta orden
6. Max costo Claude API configurable (default: $0.50/día)
7. Esto es experimental — empezar con paper trading mínimo 1 mes

### poly_probability_model.py — Modelo de probabilidad LLM (Fase 2)
Lógica:
1. Para mercados seleccionados, usar Claude haiku para estimar probabilidad real del evento
2. Comparar probabilidad estimada vs. precio actual del mercado
3. Si diferencia > 5%: tomar posición direccional
4. Fuentes para el LLM: noticias recientes, polls, datos históricos, contexto político
5. Re-evaluar cada 4-6 horas (no constantemente — controlar costos API)
6. Tracking de accuracy del modelo vs. resolución real de mercados
7. Paper trading hasta accuracy > 60% en backtesting

### cross_platform_arb.py — Arbitraje Polymarket vs Kalshi (Fase 4)
Lógica:
1. Identificar mercados equivalentes entre Polymarket y Kalshi
2. **Capa de normalización de contratos obligatoria:**
   - Comparar reglas de resolución (pueden diferir entre plataformas)
   - Verificar fechas de expiración (deben coincidir o ser equivalentes)
   - Verificar tamaños mínimos de orden en ambas plataformas
   - Solo operar pares donde la equivalencia esté 100% verificada
3. Si precio YES en Polymarket + precio NO en Kalshi < 0.97 → arbitraje
4. Considerar fees de ambas plataformas y settlement time
5. Mantener lista blanca de pares verificados manualmente
6. Solo implementar cuando capital > $5,000

## MÓDULO 3: IB PAIRS TRADING (Paper trading primero)

### ib_pairs.py — Mean reversion en pares de ADRs + ETFs
Pares a monitorear:
- ADRs argentinos: BMA/GGAL, YPF/PAMP, CEPU/LOMA
- ETFs sectoriales: XLE/XOP, XLF/KBE, EWZ/ARGT

Lógica:
1. Test de cointegración Engle-Granger (no solo correlación) — verificar que el par es realmente mean-reverting
2. Calcular z-score del spread entre cada par (ventana: 20 días)
3. Regime detection: no operar mean reversion si el par está en tendencia (rolling beta > threshold)
4. Entry: z-score > 2.0 o < -2.0, solo si cointegración es significativa (p-value < 0.05)
5. Exit: z-score cruza 0, o stop-loss en z-score > 3.5
6. Position sizing: Kelly criterion simplificado
7. Operar solo en horario de mercado US
8. No operar si correlación rolling 60d del par < 0.7

Notas IB:
- Usar `ib_async` como wrapper (pip install ib_async) — **NO usar ib_insync (archivado marzo 2024)**
- `ib_async` es el sucesor mantenido por la comunidad (ib-api-reloaded)
- Paper trading nativo: IB tiene paper account vía API (puerto 4002)
- IB Gateway para headless: más liviano que TWS, acepta API por defecto
- Puertos: 4001 (live), 4002 (paper)
- Reconexión automática si IB Gateway se reinicia
- Order types: usar LIMIT siempre
- **Fase 1: solo paper trading ($0 de capital real)**
- **Fase 3+: activar live cuando pairs estén validados y capital > $1,500**

## MÓDULO 4: TELEGRAM BOT

### telegram_bot.py
Comandos:
- `/status` → Posiciones abiertas, PnL del día, capital disponible
- `/trades` → Últimos 10 trades con PnL
- `/stop` → Kill switch manual
- `/start` → Reanudar trading
- `/risk` → Métricas de riesgo actuales (Sharpe, drawdown, fill rate)
- `/pnl [periodo]` → PnL por día/semana/mes
- `/set [param] [value]` → Cambiar config en caliente (ej: max_drawdown)
- `/audit [weekly|monthly]` → Generar reporte de auditoría
- `/compare [strategy] [period]` → Comparar performance entre períodos

Notificaciones automáticas:
- Cada trade ejecutado (ticker, dirección, precio, PnL estimado)
- Alertas de riesgo (drawdown warnings, execution degradation)
- Resumen diario a las 22:00 UTC
- Auditoría semanal automática (domingos)

## MÓDULO 5: CONFIGURACIÓN

### settings.yaml
```yaml
general:
  mode: paper  # paper | live
  capital_usd: 500
  log_level: INFO

polymarket:
  api_key: ${POLY_API_KEY}
  chain_id: 137  # Polygon
  min_profit_threshold: 0.015  # 1.5% mínimo
  ws_url: wss://ws-subscriptions-clob.polymarket.com/ws/market
  max_position_per_market: 100  # USD (ajustado a $500 capital)
  market_maker:
    target_spread: 0.05
    max_inventory: 200  # USD por mercado
    repost_threshold: 0.02  # repostear si mercado mueve > 2c

ib:
  host: 127.0.0.1
  port: 4002  # 4002 = paper, 4001 = live
  client_id: 1
  pairs:
    - [BMA, GGAL]
    - [YPF, PAMP]
    - [CEPU, LOMA]
    - [XLE, XOP]
    - [XLF, KBE]
    - [EWZ, ARGT]
  zscore_entry: 2.0
  zscore_exit: 0.0
  zscore_stop: 3.5
  cointegration_pvalue: 0.05

kalshi:  # Fase 4
  api_key: ${KALSHI_API_KEY}
  demo_mode: true

llm:
  provider: anthropic
  model: claude-haiku-4-5-20251001    # Modelo preferido
  fallback_model: claude-sonnet-4-6   # Fallback si el modelo primario no está disponible
  validate_model_on_startup: true     # Verificar disponibilidad via /v1/models al iniciar
  api_key: ${ANTHROPIC_API_KEY}
  max_cost_per_day: 0.50  # USD
  news_edge_confidence_threshold: 0.70
  probability_model_diff_threshold: 0.05

risk:
  max_daily_drawdown_pct: 5.0  # Más agresivo con $500 (5% = $25)
  max_position_pct: 20.0       # Con $500, 5% = solo $25, demasiado chico
  max_total_exposure_pct: 60.0  # Ajustado para capital chico
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

## MÓDULO 6: DEPLOYMENT

### Infraestructura

**Fase 1 (solo Polymarket — no necesita IB Gateway):**
- Puede correr local (tu PC) o VPS barato ($5/mes)
- VPS: Ubuntu 24.04 LTS, 1 vCPU, 1GB RAM, 25GB SSD (DigitalOcean/Hetzner)
- SQLite para storage (no necesita DB server separado)

**Fase 3+ (cuando se active IB live):**
- IB Gateway + Xvfb (display virtual) + IBC (auto-login) en el VPS
- Docker recomendado: usar imagen `gnzsnz/ib-gateway-docker`
- O instalar manualmente: IB Gateway + Xvfb + supervisord

### Archivos de deploy requeridos

#### docker-compose.yaml (opcional pero recomendado)
- Contenedor único con Python 3.12-slim
- Volumen persistente para SQLite y logs
- Restart policy: always
- Healthcheck: ping al engine cada 60s

#### deploy/bot.service (systemd — alternativa sin Docker)
```ini
[Unit]
Description=Trading Bot
After=network.target

[Service]
Type=simple
User=botuser
WorkingDir=/opt/trading-bot
ExecStart=/opt/trading-bot/venv/bin/python main.py
Restart=always
RestartSec=10
EnvironmentFile=/opt/trading-bot/config/secrets.env

[Install]
WantedBy=multi-user.target
```

### Monitoreo
- Telegram como canal primario de monitoreo (ya cubierto en Módulo 4)
- Heartbeat: el bot envía un ping a Telegram cada 30 min para confirmar que está vivo
- Si Telegram no recibe heartbeat en >60 min → el VPS puede enviar alerta por email vía cron
- Logs rotativos: max 100MB, retener 7 días

### Seguridad
- secrets.env con permisos 600 (solo el usuario del bot puede leer)
- Firewall: solo SSH (puerto custom, no 22) + salida HTTPS
- No exponer ningún puerto al exterior — el bot solo hace requests salientes
- IB Gateway credentials protegidos (fase 3+)

## DEPENDENCIAS (requirements.txt)

```
ib_async>=1.0.0            # Interactive Brokers (sucesor de ib_insync archivado)
py-clob-client>=0.34       # Polymarket CLOB
anthropic>=0.40            # Claude API
python-telegram-bot==20.7
aiohttp>=3.9
pyyaml>=6.0
python-dotenv>=1.0
numpy>=1.26
pandas>=2.1
scipy>=1.12
statsmodels>=0.14         # Engle-Granger cointegration test para pairs trading
aiosqlite>=0.19
feedparser>=6.0
```

## INSTRUCCIONES DE IMPLEMENTACIÓN

### Fases (respetar este orden estrictamente)

**FASE 1A — MVP Polymarket con $500 (solo market maker live):**
1. core/engine.py + risk_manager.py + portfolio.py
2. connectors/polymarket_client.py (WebSocket-first)
3. strategies/poly_market_maker.py (Avellaneda-Stoikov en mercados nicho)
4. connectors/telegram_bot.py (mínimo: notificaciones + /status + /stop)
5. data/db.py (SQLite WAL + índices)
6. Deploy local o VPS barato
7. Paper trade Polymarket 2+ semanas, luego live con $400

**FASE 1B — IB Pairs paper (en paralelo, pero risk aislado):**
8. connectors/ib_client.py (via ib_async, solo paper)
9. strategies/ib_pairs.py (paper trading en IB paper account)
10. IB pairs corre con su propio tracking de risk separado — NO mezclar decisiones de riesgo con Polymarket live
11. IB pairs en paper indefinido hasta tener capital para live

**FASE 2 — LLM (cuando fase 1 sea rentable):**
11. connectors/llm_client.py (Claude API wrapper)
12. connectors/news_scraper.py
13. strategies/poly_news_edge.py (pipeline 2 etapas)
14. strategies/poly_probability_model.py
15. Paper trade LLM strategies 1+ mes

**FASE 3 — Escalar (si 2 meses rentables):**
16. Meter más capital ($1,000-2,000)
17. Activar IB pairs live
18. Evaluar news edge live
19. Setup IB Gateway en VPS (si no estaba)

**FASE 4 — Cross-Platform Arb (solo si capital > $5,000):**
20. connectors/kalshi_client.py (demo API primero)
21. strategies/cross_platform_arb.py (con normalización de contratos)
22. Paper trade, luego live

No saltar de fase hasta que la anterior esté estable y testeada.

### Reglas de código
1. **Todo async**: usar asyncio consistentemente, no mezclar sync/async
2. **WebSocket-first**: preferir WebSocket sobre polling para market data
3. **Type hints**: en todas las funciones, usar dataclasses para modelos
4. **Error handling**: nunca crashear el bot — catch exceptions, log, notify, continue
5. **Paper trading primero**: el modo paper debe simular exactamente el modo live
6. **Cada strategy es independiente**: debe poder activarse/desactivarse sin afectar otras
7. **Logging**: cada trade, cada señal evaluada, cada error — todo a SQLite + console
8. **Sin sobre-ingeniería**: reglas claras, ejecución rápida
9. **Código conciso**: funciones cortas, nombres descriptivos, sin comentarios obvios
10. **Testing**: tests unitarios para risk_manager y cada strategy con datos mock

## MÓDULO 7: SISTEMA DE AUDITORÍAS

### Dashboard de Performance (automatizado)
El bot genera reportes vía Telegram y los guarda en SQLite.

Métricas trackeadas por estrategia:
- PnL diario/semanal/mensual (realizado + no realizado)
- Win rate y profit factor
- Sharpe ratio rolling (30 días)
- Max drawdown del período
- Número de trades y frecuencia
- Costo de Claude API
- Slippage promedio vs precio esperado
- Fill rate

### Auditoría Semanal (domingos vía Telegram)
Comando: `/audit weekly`

Checklist:
1. Performance por estrategia: cuál está funcionando y cuál no
2. Risk check: algún drawdown preocupante o exposición excesiva
3. Ejecución: trades al precio esperado, slippage
4. Costos: fees Polymarket + comisiones IB + Claude API
5. Errores: reconexiones, timeouts, fallos de ejecución

### Auditoría Mensual (revisión profunda)
Comando: `/audit monthly`

1. **Go/Kill por estrategia:**
   - Pierde dinero 2 meses consecutivos → pausar y analizar
   - Sharpe < 0.5 después de 1 mes → revisar parámetros o desactivar
   - Win rate < 45% → algo está fundamentalmente mal

2. **Rebalanceo de capital:**
   - Mover capital de estrategias underperforming a las que funcionan
   - No más del 50% del capital en una sola estrategia

3. **Calibración de parámetros:**
   - Z-score thresholds para pairs
   - Spread targets para market making
   - Confidence thresholds del LLM
   - Lista de mercados target en Polymarket

### Auditoría Trimestral (estratégica)
1. ¿El mercado cambió? Fees, restricciones, competencia, regulación
2. ¿Nuevas oportunidades? Plataformas, pares, fuentes de datos
3. ¿Escalar o pivotar?

### Tabla SQLite de auditorías
```sql
CREATE TABLE audits (
    id INTEGER PRIMARY KEY,
    date TEXT,
    type TEXT,  -- 'weekly' | 'monthly' | 'quarterly'
    strategy TEXT,
    metrics JSON,  -- {pnl, sharpe, win_rate, drawdown, trades, fees, fill_rate}
    notes TEXT,
    actions TEXT
);
```

### KPIs Operativos (objetivos)
- Sharpe ratio > 1.5
- Max drawdown < 5%
- Fill rate > 90%
- Slippage < 0.5% promedio
- PnL neto de fees positivo rolling 30 días
- Costo Claude API < $20/mes
- **Con $500: objetivo mínimo = no perder dinero los primeros 2 meses**

### Criterios de Escalado de Capital
| Condición | Acción |
|---|---|
| 2 meses rentables, Sharpe > 1.0 | Escalar a $1,000-2,000 |
| 4 meses rentables, Sharpe > 1.5 | Escalar a $5,000+, activar IB live |
| Drawdown > 10% ($50+ perdidos) | Pausar, analizar, ajustar parámetros |
| 2 meses con pérdida | Volver a paper, no meter más capital |
| Estrategia individual pierde 3 meses seguidos | Desactivar estrategia |

## INSTRUCCIONES PARA EL LLM

- Implementá módulo por módulo en orden. No generes todo junto.
- Antes de cada módulo, confirmá que entendés la lógica y preguntá si hay ambigüedades.
- Usá el mínimo de líneas posible sin sacrificar legibilidad.
- Si una dependencia tiene problemas conocidos o hay una mejor alternativa, decímelo.
- Si alguna estrategia tiene un flaw lógico, señalalo antes de implementar.
- Al terminar cada módulo, dame instrucciones para testear manualmente.
