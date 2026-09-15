# ResearchLab — SUAS Enterprises

A research platform that turns natural-language market questions ("Does buying NIFTY after a sharp fall work?") into structured, historically-tested experiments.

The workflow is **Ask → Clarify → Define → Test → Learn**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (Next.js 16 + React 19 + Tailwind 4)          │
│  ┌─────────┐  ┌──────────┐  ┌────────┐  ┌──────────┐  │
│  │  Ask     │→ │ Clarify  │→ │ Define │→ │ Results  │  │
│  │  Screen  │  │ Chat +   │  │ Review │  │ Charts + │  │
│  │          │  │ FAQs     │  │ Proto- │  │ Evidence │  │
│  └─────────┘  └──────────┘  │ col    │  └──────────┘  │
│                              └────────┘                  │
└────────────────────────┬────────────────────────────────┘
                         │ REST (JSON)
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Backend (Flask)  :8000                                 │
│  ┌──────────────────┐  ┌─────────────────────────────┐ │
│  │ /api/parse-question │  /api/backtest              │ │
│  │                   │  │                             │ │
│  │ 1. Try AI provider│  │ 1. Validate experiment      │ │
│  │ 2. Fall back to   │  │ 2. Load OHLC CSV            │ │
│  │    regex rules    │  │ 3. Run event-based engine   │ │
│  │ 3. Return parsed  │  │ 4. Explain results (AI/fallback)│
│  │    interpretation │  │ 5. Return metrics + data    │ │
│  └──────────────────┘  └─────────────────────────────┘ │
│                          │                              │
│              ┌───────────┴───────────┐                  │
│              │  AI Provider Layer    │                  │
│              │  (Gemini / Ollama /   │                  │
│              │   OpenAI-compat)      │                  │
│              └───────────────────────┘                  │
│                          │                              │
│              ┌───────────┴───────────┐                  │
│              │  Data Layer           │                  │
│              │  (NIFTY 50/250 OHLC   │                  │
│              │   CSVs via KaggleHub) │                  │
│              └───────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
```

**Request flow:**

1. **Ask** — User submits a natural-language question.
2. **Backend parse-question** — Sends the question to an AI provider (Gemini by default) with a structured JSON prompt. If AI is unavailable, falls back to a regex-based `_fallback_interpretation()` that extracts keywords (markets, thresholds, holding periods) from the question.
3. **Clarify** — Frontend shows the AI's interpretation, assumptions, and critical questions. User answers or tweaks; backend re-parses until all critical questions are resolved.
4. **Define** — Final review of the experiment protocol.
5. **Backtest** — Backend runs an event-based engine: finds all days meeting the signal condition (close-to-close fall ≥ threshold), enters at next open/close, exits after N days, deducts costs, computes metrics.
6. **Results** — Frontend renders: candlestick chart with signal markers, event queue, trade detail panel, return distribution, AI-generated evidence summary with caveats and next questions.

---

## Technology choices

| Layer | Choice | Why |
|---|---|---|
| **Frontend framework** | Next.js 16 (App Router) | Server components by default, Turbopack dev, React 19, straight-forward REST client |
| **Styling** | Tailwind CSS 4 + custom CSS | Fast iteration; the candlestick chart and stepper are hand-rolled SVG/CSS (no chart lib needed for candles) |
| **Charts** | Custom SVG (no Recharts for candles) | Full control over signal markers, crosshair, trade entry/exit dots; Recharts kept only for tiny sparkline bars |
| **Backend** | Flask (stdlib `urllib`, no Requests) | Zero micro-deps; easy to run; the only HTTP need is outbound to AI providers and inbound from the SPA |
| **Data** | Pandas + CSV | OHLC data is small (daily bars for ~10 years ≈ 2,500 rows). Pandas gives fast groupby/pct_change without a DB. |
| **Data source** | KaggleHub → local CSV | Pull once via `download_data.py`, then work offline. Avoids API rate limits during backtests. |
| **AI** | Pluggable: Gemini / Ollama / OpenAI-compat | Provider chosen via `AI_MODEL` env var. Same JSON-shape prompt works across all three. |
| **Config** | `python-dotenv` + env vars | No config files to commit; all paths/keys/providers set in `.env`. |
| **Validation** | Hand-rolled type checks | Pydantic avoided to keep dependency count at 4 small packages (flask, kagglehub, pandas, python-dotenv). |

---

## Key assumptions

These are baked into the backtest engine and surfaced to the user during the Clarify step:

1. **Signal = close-to-close one-day move.** Entry always happens on the *next* trading day (open or close, user's choice). Intraday signals are not modeled.
2. **Exit = close price after N trading days.** No intraday exits, no stop-loss, no profit target.
3. **Transaction cost is a flat % per trade** (default 0.10%). No slippage model, no market impact, no taxes.
4. **No overlapping trades.** Each signal produces exactly one trade; a new signal during an existing hold is ignored.
5. **Adjusted prices are not used.** The CSVs are raw OHLC — dividends and corporate actions are not adjusted out. (Flagged as a critical question.)
6. **Index as proxy.** NIFTY 50 / NIFTY 250 are used as research proxies, not as directly tradable instruments (no ETF tracking error modeled).
7. **"Work" = positive average net return.** Win rate and average return are the primary success metrics, not Sharpe, max drawdown, or benchmark outperformance.
8. **Minimum 25 trades for a reliable conclusion.** Below that, the UI shows an "insufficient data" warning.
9. **Historical sample ≠ forecast.** Every result page carries a caveat that past performance does not predict future outcomes.

---

## How to run the project

### Prerequisites

- Python 3.11+
- Node.js 20+
- A Kaggle account (for initial data download)
- An AI provider key (Gemini API key, or a local Ollama instance)

### 1. Backend

```bash
# from project root
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Create .env from template
cp .env.example .env               # then edit:
#   NIFTY50_DATA_CSV_PATH=data/nifty50_ohlc.csv
#   NIFTY250_DATA_CSV_PATH=data/nifty250_ohlc.csv
#   AI_PROVIDER=gemini
#   AI_MODEL=gemini-3.6-flash
#   GEMINI_API_KEY=your_key_here

# Download historical data (one-time)
python download_data.py --market nifty50
python download_data.py --market nifty250

# Run the API
python main.py                     # starts on http://localhost:8000
```

### 2. Frontend

```bash
cd frontend/suas
npm install
# .env.local:
#   NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev                        # starts on http://localhost:3000
```

Open http://localhost:3000. The frontend works in **demo mode** without the backend (shows a banner, uses synthetic data).

### 3. Run tests

```bash
# Backend tests
pytest test_api.py test_backtest.py test_assumptions.py

# Frontend type-check
cd frontend/suas && npx tsc --noEmit
```

---

## AI tools used

| Tool | Where | Purpose |
|---|---|---|
| **Gemini 3.6 Flash** (default) | `main.py` → `_ai_json()` | Interprets natural-language questions into structured experiment JSON; generates human-readable result explanations |
| **Ollama** (local) | `main.py` → `_ai_json()` | Drop-in local alternative. Set `AI_PROVIDER=ollama` and `AI_URL=http://localhost:11434/api/chat` |
| **OpenAI-compatible** | `main.py` → `_ai_json()` | Any OpenAI-compat endpoint (Groq, Together, etc.) via `AI_PROVIDER=openai` + `AI_API_URL` |

The AI layer is **optional**. If no API key is configured, or if the provider returns an error, the backend transparently falls back to a regex-based interpreter (`_fallback_interpretation`) and a template-based explainer (`_explain_results`). The frontend shows an "ai_generated: false" badge on fallback content.

**Prompt design:**
- Temperature pinned to 0.1 for deterministic JSON output.
- Gemini uses `responseMimeType: application/json`; Ollama uses `format: "json"`; OpenAI uses `response_format: {"type": "json_object"}`.
- A retry loop re-calls the API once if the first response isn't valid JSON, then gives up to the fallback.

---

## What we would improve next

1. **Statistical rigor.** The current conclusion is "average return ≥ 0 → positive." Next: t-test / bootstrap confidence intervals, Sharpe ratio, maximum drawdown, and out-of-sample testing (walk-forward).

2. **Overlapping trades & position sizing.** Currently one signal = one trade, no overlaps. A real system would model capital allocation, pyramiding, and concurrent positions.

3. **Adjusted prices.** Use total-return indices or adjust for dividends. Raw OHLC understates long-term returns.

4. **Slippage & market impact model.** A flat 0.10% cost is a rough proxy. Next: volume-based slippage, bid-ask spread estimates.

5. **More signal types.** Currently only "sharp fall reversal." Add breakouts, moving-average crossovers, volatility contractions, multi-day patterns.

6. **Benchmark comparison.** Show alpha vs. buy-and-hold, not just raw returns. The backend already computes `benchmark_return_pct`; the frontend doesn't visualize it yet.

7. **Persistence.** Experiments and results are ephemeral (in-memory state on the frontend). Next: save to a database, shareable links, experiment history.

8. **Authentication & multi-user.** Currently single-tenant, no auth. For team use: login, saved experiments, role-based access.

9. **Production deployment.** The Flask dev server is not production-grade. Next: Gunicorn + reverse proxy, Docker image, CI pipeline.

10. **Better AI fallback.** The regex fallback handles common patterns but misses nuance. A small fine-tuned model (or a stronger prompt with few-shot examples) would close the gap when the primary provider is unavailable.
