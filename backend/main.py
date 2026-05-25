import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

MOONDREAM_API_KEY = os.getenv("MOONDREAM_API_KEY", "")
MOONDREAM_BASE_URL = "https://api.moondream.ai"
MOONDREAM_MODEL = "moondream3-preview"

app = FastAPI(title="Face Analyzer Proxy", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Shared HTTP client (connection-pooled, reused across requests) ──
_http = httpx.AsyncClient(timeout=120)


# ─── Models ──────────────────────────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    image_b64: str  # raw base64, no data-URL prefix
    prompt: str


# ─── Groq helper ─────────────────────────────────────────────────────────────


async def _call_groq(image_b64: str, prompt: str) -> str:
    """Call Groq chat-completions with a vision model (OpenAI-compatible)."""
    resp = await _http.post(
        f"{GROQ_BASE_URL}/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 1300,
            "temperature": 0.3,  # lower temperature for more factual/concise answers, adjust as needed
            "stream": False,
            "top_p": 1,
        },
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    if not content:
        raise ValueError("Empty response from Groq")
    return content


# ─── Moondream helper (legacy fallback) ───────────────────────────────────────


async def _call_moondream(image_b64: str, question: str) -> str:
    """Call Moondream3 /v1/query and return the answer string."""
    resp = await _http.post(
        f"{MOONDREAM_BASE_URL}/v1/query",
        headers={
            "Content-Type": "application/json",
            "X-Moondream-Auth": MOONDREAM_API_KEY,
        },
        json={
            "image_url": f"data:image/jpeg;base64,{image_b64}",
            "question": question,
            "model": MOONDREAM_MODEL,
        },
    )
    resp.raise_for_status()
    payload = resp.json()
    answer = payload.get("answer") or payload.get("text") or ""
    if not answer:
        raise ValueError(f"Empty answer from Moondream: {payload}")
    return answer


# ─── Claude helper ────────────────────────────────────────────────────────────


async def _call_claude(image_b64: str, prompt: str) -> str:
    """Call Claude via Anthropic /v1/messages and return the text content."""
    resp = await _http.post(
        f"{ANTHROPIC_BASE_URL}/v1/messages",
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": ANTHROPIC_VERSION,
        },
        json={
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1300,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        },
    )
    resp.raise_for_status()
    payload = resp.json()
    return "".join(
        block["text"]
        for block in payload.get("content", [])
        if block.get("type") == "text"
    )


# ─── Unified analyze endpoint ─────────────────────────────────────────────────


@app.post("/v1/analyze")
async def analyze(req: AnalyzeRequest):
    """
    Priority chain: Groq (Llama vision) → Moondream → Claude.
    Returns { answer: str, model: str }.
    """
    errors = []

    # ── 1. Groq / Llama vision (primary) ──
    if GROQ_API_KEY:
        try:
            answer = await _call_groq(req.image_b64, req.prompt)
            return JSONResponse({"answer": answer, "model": GROQ_MODEL})
        except Exception as exc:
            errors.append(f"Groq: {exc}")
            print(f"[Groq] failed ({type(exc).__name__}: {exc}), trying next")

    # ── 2. Moondream (secondary) ──
    if MOONDREAM_API_KEY:
        try:
            answer = await _call_moondream(req.image_b64, req.prompt)
            return JSONResponse({"answer": answer, "model": MOONDREAM_MODEL})
        except Exception as exc:
            errors.append(f"Moondream: {exc}")
            print(f"[Moondream] failed ({type(exc).__name__}: {exc}), trying next")

    # ── 3. Claude (final fallback) ──
    if not ANTHROPIC_API_KEY:
        raise HTTPException(
            status_code=503,
            detail=(
                "No AI backend configured. "
                "Set GROQ_API_KEY, MOONDREAM_API_KEY, or ANTHROPIC_API_KEY in backend/.env"
            ),
        )

    try:
        answer = await _call_claude(req.image_b64, req.prompt)
        return JSONResponse({"answer": answer, "model": "claude-sonnet-4"})
    except Exception as exc:
        errors.append(f"Claude: {exc}")
        raise HTTPException(status_code=502, detail=" | ".join(errors))


# ─── Health ───────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    if GROQ_API_KEY:
        primary = GROQ_MODEL
    elif MOONDREAM_API_KEY:
        primary = MOONDREAM_MODEL
    elif ANTHROPIC_API_KEY:
        primary = "claude-sonnet-4"
    else:
        primary = "none"
    return {
        "status": "ok",
        "primary_model": primary,
        "groq_key_set": bool(GROQ_API_KEY),
        "moondream_key_set": bool(MOONDREAM_API_KEY),
        "anthropic_key_set": bool(ANTHROPIC_API_KEY),
    }


# ─── Frontend static files ────────────────────────────────────────────────────
# Mounted last so API routes above always take precedence.
# html=True serves index.html for "/" and any unmatched path (SPA fallback).
_FRONTEND = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=_FRONTEND, html=True), name="frontend")


# ─── Frontend index ───────────────────────────────────────────────────────────


@app.get("/")
async def read_index():
    return FileResponse(_FRONTEND / "index.html")