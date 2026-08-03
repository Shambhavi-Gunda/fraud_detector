# Fraudulent Booking Detector

An LLM-based agent that investigates travel booking transactions for fraud
signals, exposed as a FastAPI service. The model drives an agentic tool-use
loop: it decides which investigative tools to call (geo mismatch, account
history, booking velocity, price anomaly, email reputation, booking pattern),
inspects the results, and only then produces a structured risk verdict.

Supports two free-tier LLM providers — switch with one env var:
- **Groq** (default) — fast open-weight models (Llama 3.3 70B), generous free tier, no credit card.
- **Gemini** — Google's Gemini 2.5 Flash, also has a free tier, no credit card.

## Project layout

```
fraud_detector/
├── models.py            # Pydantic schemas: BookingTransaction in, FraudVerdict out
├── synthetic_data.py     # Generates a realistic synthetic booking dataset w/ seeded fraud patterns
├── tools.py               # The "investigation tools" the agent can call + their tool schemas
├── agent.py                # The agentic loop: FraudAgent.analyze(transaction) -> FraudVerdict
│                              (supports provider="groq" or provider="gemini")
├── main.py                  # FastAPI app wiring it all together
├── test_agent_live.py        # Small script to sanity-check the live agent on 2 sample transactions
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

### Get a free API key

**Groq** (default, recommended for speed): https://console.groq.com → sign up, no
credit card required → create an API key (starts with `gsk_...`).

**Gemini**: https://aistudio.google.com/apikey → sign in with Google, no credit
card required → create an API key (starts with `AIza...`).

You only need ONE of these, matching whichever provider you choose below.

### Configure

```bash
# Choose a provider (defaults to "groq" if unset)
export LLM_PROVIDER=groq          # or: export LLM_PROVIDER=gemini

# Set the matching key
export GROQ_API_KEY=gsk_...       # if using groq
export GEMINI_API_KEY=AIza...     # if using gemini
```

Optional overrides:
```bash
export GROQ_MODEL=llama-3.3-70b-versatile   # default
export GEMINI_MODEL=gemini-2.5-flash         # default
```

## Try the agent directly (no server)

```bash
python3 test_agent_live.py
```

This generates a small synthetic dataset, then runs the live agent against
one seeded-fraud and one seeded-legit transaction and prints the reasoning,
flags, and verdict.

## Run the API

```bash
uvicorn main:app --reload --port 8000
```

Then, e.g.:

```bash
# 1. Generate a synthetic dataset of 200 bookings, ~12% seeded as fraud
curl -X POST localhost:8000/transactions/generate \
  -H "Content-Type: application/json" \
  -d '{"n": 200, "fraud_rate": 0.12, "seed": 42}'

# 2. See what got loaded
curl "localhost:8000/transactions?limit=5"

# 3. Analyze one specific transaction by booking_id (from step 2's output)
curl -X POST localhost:8000/analyze/<booking_id>

# 4. Analyze a fully custom transaction you construct yourself
curl -X POST localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d @sample_transaction.json

# 5. Batch-analyze everything currently loaded, with a precision/recall
#    summary against the synthetic ground-truth labels
curl -X POST localhost:8000/analyze/batch -H "Content-Type: application/json" -d '{}'
```

Interactive API docs: http://localhost:8000/docs

## A note on free-tier rate limits

Both providers' free tiers are rate-limited (requests per minute and per
day), and each transaction the agent investigates can involve several tool
calls, i.e. several API round trips. The agent automatically retries with
exponential backoff on 429 (rate limit) errors, but if you hit `/analyze/batch`
with a large dataset on a free key, expect it to slow down or need multiple
attempts. For quick testing, keep batches small (10-30 transactions) or use
`/analyze/{booking_id}` on individual bookings.

Roughly, as of this writing: Groq's free tier is generous on requests/day but
tight on requests/minute for the larger models; Gemini 2.5 Flash's free tier
is ~10 requests/minute, 1,500/day. Check your provider's current console for
exact limits, since these change.

## How detection works

**Synthetic data (`synthetic_data.py`)** seeds realistic fraud patterns:
new accounts, IP/billing/card-BIN country mismatches, last-minute one-way
international bookings, disposable email domains, prior chargebacks, and
"device farming" clusters (one device fingerprint used across several
different user identities in quick succession — a classic bot/fraud-ring
signature).

**Tools (`tools.py`)** are the agent's investigative capabilities:
- `check_geo_mismatch` — IP vs. billing vs. card-BIN country agreement
- `check_email_reputation` — disposable/throwaway domain detection
- `check_account_history` — account age, booking history, chargebacks
- `check_price_anomaly` — deviation from a route's reference price
- `check_velocity` — device fingerprint reuse across identities
- `check_booking_pattern` — last-minute / one-way / same-day red flags

In production, swap the bodies of these functions for real calls to a geo-IP
service, your bookings database, a payments risk API, and an email
reputation service — the tool *schemas* the model sees don't need to change.

**Agent (`agent.py`)** runs a real tool-use loop (not a fixed pipeline): the
model reads the transaction, decides which tools are relevant, reads their
results, optionally calls more tools, then returns a single structured JSON
verdict: `risk_score` (0–100), `risk_level`, specific `flags`, a
human-readable `explanation`, and a `recommended_action`
(`approve` / `approve_with_monitoring` / `manual_review` /
`hold_for_verification` / `decline`).

Under the hood, `FraudAgent` has one method per provider (`_run_groq`,
`_run_gemini`) since Groq (OpenAI-style `tool_calls`) and Gemini
(`function_call` parts in `Content` objects) use different tool-calling
formats — but both converge on the same `FraudVerdict` output, so the rest
of the app (`main.py`, `tools.py`, `models.py`) doesn't care which provider
is active.

## Notes on the synthetic data

Each generated record includes a `label` field (`"fraud"` or `"legit"`).
This is **ground truth for evaluation only** — `main.py` strips it out
before sending the transaction to the agent, and `/analyze/batch` uses it
afterward purely to compute a true/false positive/negative summary. The
agent never sees it.

## Extending this

- **Real data**: point `FraudAgent(dataset=...)` at your real bookings table
  instead of `synthetic_data.py`'s output, and wire the tool functions in
  `tools.py` to real lookups.
- **Persistence**: swap the in-memory `_state` dict in `main.py` for a real
  database.
- **Human-in-the-loop**: `manual_review` / `hold_for_verification` verdicts
  are natural queues to route into a review dashboard rather than
  auto-approving or auto-declining.
- **Another provider**: add a new `_run_<provider>` method to `FraudAgent`
  following the pattern of `_run_groq` / `_run_gemini`, and add it to the
  branch in `__init__` and `analyze()`.
