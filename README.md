# Face Analyzer

Real-time facial analysis in the browser powered by MediaPipe Tasks Vision, with AI-driven deep analysis via Groq (Meta-Llama 4 Scout 17B 16E Instruct) or Moondream.

---

## Features

**Live sidebar (every frame)**

| Feature | Model |
|---|---|
| Face detection | BlazeFace Short-Range |
| Ornament detection | AI model |
| Camera distance (IPD%) | Iris landmarks 468 / 473 |

**Deep Analysis modal (on demand)**

| Feature | Source |
|---|---|
| Overview | AI (vision model) |
| Ornament detection | AI model |
| Skin undertone (Cool / Warm / Neutral) | ITA formula on CIE LAB + vein pixel analysis |
| Metal recommendations (gold, silver, etc.) | Undertone lookup table (deterministic) |
| Complementary colors | Undertone lookup table (deterministic) |
| Makeup recommendations | AI |
| Skincare note | AI |

---

## Stack

- **Frontend** — Vanilla JS, MediaPipe Tasks Vision (`@mediapipe/tasks-vision`), zero build step
- **Backend** — FastAPI + httpx, proxies requests to AI vision APIs
- **Primary AI** — [Groq](https://console.groq.com) — Meta-Llama 4 Scout 17B 16E Instruct (free tier, fast)
- **Secondary AI** — [Moondream3-preview](https://api.moondream.ai) (fallback)
- **Fallback AI** — Claude Sonnet (Anthropic)

---

## Setup

```bash
# 1. Install backend dependencies
pip install -r backend/requirements.txt

# 2. Configure environment
cp backend/.env.example backend/.env   # fill in your API keys

# 3. Run
python main.py
```

Open `http://localhost:8000` — the backend serves the frontend and all APIs on the same port.

> To point at a different backend (e.g. a deployed instance), set `BACKEND_URL` at the top of `frontend/js/script.js`.

---

## Environment Variables

Create `backend/.env` from the provided example:

```env
# Primary — Groq (Meta-Llama 4 Scout 17B 16E Instruct, free tier)
GROQ_API_KEY=gsk_...
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct

# Secondary — Moondream3 (optional fallback)
MOONDREAM_API_KEY=your-moondream-api-key

# Fallback — Claude (used when all above are absent or fail)
ANTHROPIC_API_KEY=sk-ant-...
```

All keys are optional independently — the backend tries Groq → Moondream → Claude in order and uses whichever is configured. A `GET /health` endpoint reports which keys are set and which model is active.

Get your API keys from:
- Groq: [https://console.groq.com/keys](https://console.groq.com/keys)
- Moondream3-preview: [https://moondream.ai/me/api-keys](https://moondream.ai/me/api-keys)
- Claude: [https://console.anthropic.com](https://console.anthropic.com)

---

## Project Structure

```
face-analyzer/
├── main.py                 # Entry point — run this from the project root
├── frontend/
│   ├── index.html          # App shell + sidebar cards
│   ├── css/styles.css
│   └── js/
│       ├── imports.js      # ES module — hoists MediaPipe classes onto window
│       └── script.js       # All analysis + deep-analysis logic
├── backend/
│   ├── main.py             # FastAPI proxy (Groq + Moondream + Claude)
│   ├── requirements.txt
│   └── .env.example
└── data/
    └── models/             # Local model cache (git-ignored)
```