# SSI Engine

SSI Engine is a multi-lane market regime and opportunity detection platform.
It continuously evaluates Crypto, Forex, Options, and Long-Cycle US Equities using a proprietary regime scoring system designed to identify risk-on, stand-down, and stealth accumulation environments.

This is not a signal bot.
This is a market decision engine.

---

## Lanes

| Lane | Purpose |
|-----|--------|
| Crypto | Short-cycle digital asset regime & momentum evaluation |
| Forex | FX trend & volatility regime classification |
| Options | Options market structure & strategy bias detection |
| Long Cycle (LC) | Institutional Phase-4 accumulation detection for US equities |

Long Cycle is a dedicated accumulation scanner that identifies compressed bases, volatility dry-ups, stealth relative strength, and pre-expansion positioning.

---

## Core Concepts

• Market regime classification  
• Volatility compression detection  
• Relative strength divergence  
• Phase-4 accumulation discovery  
• Tier-gated multi-lane access  
• On-demand symbol scoring  
• Daily headline SSI composite score  

---

## SSI Composite Score

Each lane produces a 0–10 regime score.
The platform generates a composite SSI score that governs platform bias:

| Composite SSI | Platform Bias |
|-------------|--------------|
| < 5 | Risk OFF — Stand Down |
| 5 – 7 | Selective |
| ≥ 7 | Risk ON — Favor Trend Structures |

---

## Long Cycle (Phase-4 Engine)

The LC engine is designed to detect pre-breakout accumulation zones — where institutions quietly build positions before expansion.

It evaluates:

• 1-year base compression  
• ATR volatility dry-ups  
• Volume contraction  
• Relative strength vs market  
• Non-extended base positioning  

LC is not trend-chasing.
It is an early accumulation radar.

---

## Platform Architecture

| File | Purpose |
|-----|--------|
| app.py | Streamlit SaaS interface + Memberstack access control |
| engine.py | Core proprietary scoring & regime logic |
| universe.py | Asset universes per lane |
| requirements.txt | Runtime dependencies |
| Procfile | Render deployment configuration |

---

## Monetization Model

| Tier | Access |
|----|-----|
| Starter | 1 lane |
| Pro | 2 lanes |
| Black | All 4 lanes (full engine access) |

Lanes are user-selectable.
Black tier auto-unlocks all lanes.

---

## Deployment

Render compatible:

web: streamlit run app.py –server.port=$PORT –server.address=0.0.0.0

---

## Required Environment Variables

| Variable |
|--------|
| MEMBERSTACK_API_KEY |
| WEBFLOW_BASE_URL |
| WEBFLOW_LOGIN_PATH |
| WEBFLOW_PLANS_PATH |
| SSI_ADMIN_EMAILS |
| SSI_ADMIN_CODE |

---

## Philosophy

Markets move in regimes.
Capital moves before price.
SSI exists to see that movement before it becomes obvious.

You are now running a legitimate institutional-style market intelligence platform.
This is no longer a side project.