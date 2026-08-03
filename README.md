# Fraudulent Booking Detector

An LLM-powered agent that investigates travel booking transactions for fraud
signals, exposed as a FastAPI service. The model runs a real agentic
tool-use loop — it decides which investigative tools to call, reads the
results, calls more if needed, and only then returns a structured risk
verdict. It is not a fixed if/else rules engine with an LLM bolted on.

Supports two free-tier LLM providers, switchable with one environment
variable:
- **Groq** (default) — fast open-weight models (Llama 3.3 70B), generous free tier, no credit card.
- **Gemini** — Google's Gemini models, also free, no credit card.

---

## What it does

1. **Generates synthetic booking data** (`synthetic_data.py`) with realistic
   fraud patterns seeded in: brand-new accounts, IP/billing/card-country
   mismatches, last-minute one-way international bookings, disposable email
   addresses, prior chargebacks, and "device farming" (one device used
   across several different identities in quick succession).

2. **Gives the agent investigative tools** (`tools.py`) it can call to check
   a transaction:
   - `check_geo_mismatch` — IP vs. billing vs. card-BIN country agreement
   - `check_email_reputation` — disposable/throwaway email domains
   - `check_account_history` — account age, booking history, chargebacks
   - `check_price_anomaly` — deviation from a route's typical price
   - `check_velocity` — device fingerprint reuse across identities
   - `check_booking_pattern` — last-minute / one-way / same-day red flags

3. **Runs the agent** (`agent.py`) — the LLM reads a transaction, decides
   which tools to call, inspects results, and returns a structured verdict:
   `risk_score` (0–100), `risk_level` (low/medium/high/critical), specific
   `flags`, a human-readable `explanation`, and a `recommended_action`
   (approve / approve_with_monitoring / manual_review /
   hold_for_verification / decline).

4. **Serves it all as an API** (`main.py`) so you can generate data, list
   transactions, and analyze them (individually or in batch, with an
   accuracy summary against ground truth) over HTTP.

## Project layout

```
fraud_detector/
├── models.py             # Pydantic schemas: BookingTransaction in, FraudVerdict out
├── synthetic_data.py      # Generates the synthetic booking dataset
├── tools.py                # The agent's investigative tools + their schemas
├── agent.py                 # The agentic loop (Groq and Gemini implementations)
├── main.py                   # FastAPI app
├── test_agent_live.py         # Quick script: run the agent on 2 sample transactions
├── list_gemini_models.py       # Lists which Gemini models your API key can use
├── sample_transaction.json      # A deliberately suspicious example transaction
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup (Windows / VS Code)

### 1. Open the folder in VS Code
`File → Open Folder...` → select the `fraud_detector` folder.

### 2. Create a virtual environment using Python 3.12
Open a terminal in VS Code (`` Ctrl+` ``):
```powershell
py -3.12 -m venv venv
```
> Use Python 3.12 or 3.11. Newer versions (e.g. 3.14) don't have prebuilt
> wheels for some dependencies yet and will fail to install.

### 3. Activate it
```powershell
venv\Scripts\Activate.ps1
```
Your prompt should now start with `(venv)`.

### 4. Point VS Code at this environment
`Ctrl+Shift+P` → "Python: Select Interpreter" → choose `.\venv\Scripts\python.exe`.

### 5. Install dependencies
```powershell
pip install -r requirements.txt
```

### 6. Get a free API key
Pick one provider:
- **Groq**: https://console.groq.com → sign up, no card → create key (`gsk_...`)
- **Gemini**: https://aistudio.google.com/apikey → sign in with Google → create key (`AIza...`)

### 7. Create a `.env` file
In VS Code Explorer: right-click the folder → New File → name it exactly `.env`.

For Groq:
```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_actual_key_here
```

For Gemini:
```
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSy_your_actual_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
```
> Gemini's model lineup changes often, and some models 404 or return zero
> free quota depending on your account. Run `python list_gemini_models.py`
> (with `GEMINI_API_KEY` set) to see exactly which models your key can use
> if `gemini-3.1-flash-lite` ever stops working.

`.env` is already excluded via `.gitignore` — your key will never get
committed to git.

---

## Running it

### Quick test — no server, just the agent
```powershell
$env:LLM_PROVIDER="groq"          # or "gemini"
$env:GROQ_API_KEY="gsk_..."       # or $env:GEMINI_API_KEY="AIza..."
python test_agent_live.py
```
Generates a small dataset and runs the live agent on one seeded-fraud and
one seeded-legit transaction, printing risk score, flags, explanation, and
which tools it called.

### Full API service
```powershell
uvicorn main:app --reload --port 8000
```
Leave this running. Then open **Chrome** (not VS Code's built-in Simple
Browser) and go to:
```
http://127.0.0.1:8000/docs
```
This is an interactive Swagger UI — click any endpoint → "Try it out" → Execute.

Typical flow:
1. `POST /transactions/generate` — generate synthetic bookings, e.g.
   `{"n": 20, "fraud_rate": 0.15, "seed": 1}` (keep batches small on a free
   API key to avoid rate limits)
2. `GET /transactions` — see what was generated, copy a `booking_id`
3. `POST /analyze/{booking_id}` — paste the id, run it, see the agent's live verdict
4. `POST /analyze` — analyze a fully custom transaction (try pasting the
   contents of `sample_transaction.json`)
5. `POST /analyze/batch` — analyze everything currently loaded, with a
   precision/recall summary against the synthetic ground truth

To stop the server: click into the terminal, press `Ctrl+C`.

---

## A note on free-tier rate limits

Each transaction can involve several tool calls, i.e. several API round
trips. The agent automatically retries with exponential backoff on
rate-limit (429) and overload (503) errors. On a free key, keep test
batches small (10–30 transactions) and expect `/analyze/batch` on a large
dataset to be slow or need retries.

---

## Pushing this to GitHub

You're working inside a `venv`, which is good — it's already excluded from
git via `.gitignore`, along with `.env` (your API key) and `__pycache__`.

### 1. Make sure you're NOT inside the venv's own folder
Run these commands from `D:\fraud_detector` (the project root), not from
inside `venv\`. Being *activated* into the venv (prompt shows `(venv)`) is
fine and expected — that's different from being *inside* the `venv` folder.

### 2. Initialize git (skip if you've already done this once)
```powershell
git init
```

### 3. Check what will be committed
```powershell
git status
```
You should **not** see `venv/` or `.env` in the list. If you do, double
check `.gitignore` is saved in the project root and named exactly `.gitignore`
(not `.gitignore.txt` — Windows sometimes hides the real extension; enable
"File name extensions" in File Explorer's View tab to confirm).

### 4. Stage and commit
```powershell
git add .
git commit -m "Fraud detection agent with Groq/Gemini support"
```

### 5. Create a GitHub repo
Go to https://github.com/new, create an empty repository (don't
initialize it with a README — you already have one), and copy the repo URL
it gives you, e.g. `https://github.com/yourname/fraud-detector.git`.

### 6. Connect your local repo to GitHub and push
```powershell
git branch -M main
git remote add origin https://github.com/yourname/fraud-detector.git
git push -u origin main
```
The first push may prompt you to sign in to GitHub in your browser —
follow that flow if so.

### 7. Future changes
After the first push, subsequent updates are just:
```powershell
git add .
git commit -m "describe what changed"
git push
```

### If `git` isn't recognized
Install it from https://git-scm.com/download/win, restart VS Code's
terminal afterward, then retry step 2.

---

## Extending this

- **Real data**: point `FraudAgent(dataset=...)` at your real bookings
  table instead of `synthetic_data.py`'s output, and wire `tools.py`'s
  functions to real lookups (a geo-IP service, your bookings DB, a payments
  risk API, an email reputation service).
- **Persistence**: swap the in-memory `_state` dict in `main.py` for a real
  database.
- **Human-in-the-loop**: route `manual_review` / `hold_for_verification`
  verdicts into a review dashboard instead of auto-approving/declining.
- **Another provider**: add a `_run_<provider>` method to `FraudAgent`
  following the pattern of `_run_groq` / `_run_gemini`.
