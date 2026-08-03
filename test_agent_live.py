"""
Quick manual test: generates a small synthetic dataset, then runs the live
agent (real LLM API call, via Groq or Gemini) against one seeded-fraud and
one seeded-legit transaction, printing the verdicts.

Usage:
    export LLM_PROVIDER=groq          # or "gemini"
    export GROQ_API_KEY=gsk_...       # if using groq
    export GEMINI_API_KEY=AIza...     # if using gemini
    python3 test_agent_live.py
"""
import os

from agent import FraudAgent
from models import BookingTransaction
from synthetic_data import generate_dataset

if __name__ == "__main__":
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    print(f"Using provider: {provider}")

    data = generate_dataset(n=60, fraud_rate=0.15, seed=7)
    agent = FraudAgent(dataset=data, provider=provider)

    fraud_record = next(r for r in data if r["label"] == "fraud")
    legit_record = next(r for r in data if r["label"] == "legit")

    for label, record in [("SEEDED FRAUD", fraud_record), ("SEEDED LEGIT", legit_record)]:
        txn = BookingTransaction(**{k: v for k, v in record.items() if k != "label"})
        verdict = agent.analyze(txn)
        print(f"\n=== {label} — ground truth: {record['label']} ===")
        print(f"risk_score: {verdict.risk_score}  risk_level: {verdict.risk_level.value}")
        print(f"flags: {verdict.flags}")
        print(f"explanation: {verdict.explanation}")
        print(f"recommended_action: {verdict.recommended_action}")
        print(f"tool_calls made: {[t.tool for t in verdict.tool_calls]}")
