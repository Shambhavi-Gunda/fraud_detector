"""
FastAPI service for the Fraudulent Booking Detector.

Endpoints:
  GET  /                             - the web UI (static/index.html)
  GET  /health                      - liveness + provider status check
  POST /transactions/generate       - generate & load a synthetic dataset
  GET  /transactions                - list currently loaded transactions
  POST /analyze                     - analyze a single transaction (agent)
  POST /analyze/batch               - analyze all currently loaded transactions
  POST /analyze/{booking_id}        - analyze one transaction from the loaded set

Run:
  export LLM_PROVIDER=groq          # or "gemini"
  export GROQ_API_KEY=gsk_...       # if using groq
  export GEMINI_API_KEY=AIza...     # if using gemini
  uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import FraudAgent
from models import BookingTransaction, FraudVerdict
from synthetic_data import generate_dataset

load_dotenv()

app = FastAPI(
    title="Fraudulent Booking Detector",
    description="LLM agent that investigates travel booking transactions for fraud signals.",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# In-memory store standing in for a real bookings database.
_state: dict = {"dataset": []}


def get_agent() -> FraudAgent:
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    required_key = "GROQ_API_KEY" if provider == "groq" else "GEMINI_API_KEY"
    if not os.environ.get(required_key):
        raise HTTPException(
            status_code=500,
            detail=f"LLM_PROVIDER is '{provider}' but {required_key} is not set. "
            f"Export it, or set LLM_PROVIDER to switch providers.",
        )
    return FraudAgent(dataset=_state["dataset"], provider=provider)


class GenerateRequest(BaseModel):
    n: int = 200
    fraud_rate: float = 0.12
    seed: Optional[int] = 42


class BatchAnalyzeRequest(BaseModel):
    limit: Optional[int] = None  # cap how many of the loaded transactions to analyze


@app.get("/")
def serve_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    required_key = "GROQ_API_KEY" if provider == "groq" else "GEMINI_API_KEY"
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile") if provider == "groq" \
        else os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
    return {
        "status": "ok",
        "loaded_transactions": len(_state["dataset"]),
        "provider": provider,
        "model": model,
        "provider_configured": bool(os.environ.get(required_key)),
    }


@app.post("/transactions/generate")
def generate_transactions(req: GenerateRequest):
    data = generate_dataset(n=req.n, fraud_rate=req.fraud_rate, seed=req.seed)
    _state["dataset"] = data
    n_fraud = sum(1 for r in data if r["label"] == "fraud")
    return {
        "generated": len(data),
        "seeded_fraud_count": n_fraud,
        "seeded_legit_count": len(data) - n_fraud,
        "note": "The 'label' field is ground truth for evaluation only — it is never shown to the agent.",
    }


@app.get("/transactions")
def list_transactions(limit: int = 50):
    return {"count": len(_state["dataset"]), "transactions": _state["dataset"][:limit]}


@app.post("/analyze", response_model=FraudVerdict)
def analyze_transaction(txn: BookingTransaction):
    agent = get_agent()
    return agent.analyze(txn)


# IMPORTANT: /analyze/batch must be defined BEFORE /analyze/{booking_id}.
# FastAPI matches routes in definition order, so if the dynamic
# {booking_id} route came first, a request to /analyze/batch would match
# it instead, treating "batch" as a literal booking_id and 404-ing.
@app.post("/analyze/batch")
def analyze_batch(req: BatchAnalyzeRequest):
    agent = get_agent()
    records = _state["dataset"][: req.limit] if req.limit else _state["dataset"]
    if not records:
        raise HTTPException(status_code=400, detail="No transactions loaded. Call /transactions/generate first.")

    # Analyze sequentially with a small delay between calls. Each single
    # analysis can involve several LLM calls (tool-use loop), so free-tier
    # per-minute rate limits are easy to hit on a batch of many bookings.
    # A failure on one booking is recorded, not fatal to the whole batch.
    results = []
    for i, record in enumerate(records):
        txn = BookingTransaction(**{k: v for k, v in record.items() if k != "label"})
        try:
            verdict = agent.analyze(txn)
            results.append({"ground_truth_label": record["label"], "verdict": verdict, "error": None})
        except Exception as e:  # noqa: BLE001 - provider SDKs raise different exception types
            results.append({"ground_truth_label": record["label"], "verdict": None, "error": str(e)})
        if i < len(records) - 1:
            time.sleep(3)  # stay under free-tier requests-per-minute limits

    # Quick evaluation summary against the synthetic ground truth (successful analyses only).
    flagged_risk_levels = {"high", "critical"}
    succeeded = [r for r in results if r["verdict"] is not None]
    failed_count = len(results) - len(succeeded)
    tp = sum(1 for r in succeeded if r["ground_truth_label"] == "fraud" and r["verdict"].risk_level in flagged_risk_levels)
    fn = sum(1 for r in succeeded if r["ground_truth_label"] == "fraud" and r["verdict"].risk_level not in flagged_risk_levels)
    fp = sum(1 for r in succeeded if r["ground_truth_label"] == "legit" and r["verdict"].risk_level in flagged_risk_levels)
    tn = sum(1 for r in succeeded if r["ground_truth_label"] == "legit" and r["verdict"].risk_level not in flagged_risk_levels)

    return {
        "analyzed": len(results),
        "succeeded": len(succeeded),
        "failed": failed_count,
        "evaluation": {"true_positives": tp, "false_negatives": fn, "false_positives": fp, "true_negatives": tn},
        "results": results,
    }


@app.post("/analyze/{booking_id}", response_model=FraudVerdict)
def analyze_loaded_transaction(booking_id: str):
    record = next((r for r in _state["dataset"] if r["booking_id"] == booking_id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="booking_id not found in loaded dataset")
    txn = BookingTransaction(**{k: v for k, v in record.items() if k != "label"})
    agent = get_agent()
    return agent.analyze(txn)