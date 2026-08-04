# Fraudulent Booking Detector

An LLM-powered agent that investigates travel booking transactions for fraud
signals, with a polished web UI and a FastAPI backend. The model runs a real
agentic tool-use loop — it decides which investigative tools to call, reads
the results, calls more if needed, and only then returns a structured risk
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

4. **Serves a full web UI** (`static/index.html`) — generate synthetic
   bookings, click into any one to trigger a live agent investigation, and
   see the verdict as a risk gauge, flag tags, a plain-language explanation,
   the full investigation trace (which tools fired and what they found), and
   an ink-stamp-style approve/decline mark. Batch mode analyzes everything
   loaded at once and scores the agent against ground truth.

5. **Serves it all as an API too** (`main.py`), if you'd rather call it
   programmatically. Interactive API docs at `/docs`.

## Project layout

```
fraud_detector/
├── static/
│   └── index.html          # The web UI (single file, no build step)
├── models.py                # Pydantic schemas: BookingTransaction in, FraudVerdict out
├── synthetic_data.py         # Generates the synthetic booking dataset
├── tools.py                    # The agent's investigative tools + their schemas
├── agent.py                     # The agentic loop (Groq and Gemini implementations)
├── main.py                       # FastAPI app — serves both the UI and the API
├── test_agent_live.py             # Quick script: run the agent on 2 sample transactions
├── list_gemini_models.py           # Lists which Gemini models your API key can use
├── sample_transaction.json          # A deliberately suspicious example transaction
├── requirements.txt
├── Dockerfile                        # For Render / Hugging Face Spaces deployment
├── .dockerignore
├── render.yaml                        # Render one-click blueprint (native Python deploy)
├── .gitignore
└── README.md
```

---

## Local setup (Windows / VS Code)

### 1. Open the folder in VS Code
`File → Open Folder...` → select the `fraud_detector` folder.

### 2. Create a virtual environment using Python 3.12
```powershell
py -3.12 -m venv venv
```
> Use Python 3.12 or 3.11. Newer versions (e.g. 3.14) don't have prebuilt
> wheels for some dependencies yet and will fail to install.

### 3. Activate it
```powershell
venv\Scripts\Activate.ps1
```

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

## Running it locally

### Quick test — no server, just the agent
```powershell
$env:LLM_PROVIDER="groq"          # or "gemini"
$env:GROQ_API_KEY="gsk_..."       # or $env:GEMINI_API_KEY="AIza..."
python test_agent_live.py
```

### Full app (UI + API)
```powershell
uvicorn main:app --reload --port 8000
```
Leave this running. Open **Chrome** (not VS Code's built-in Simple Browser)
and go to:
```
http://127.0.0.1:8000/
```
That's the web UI. Interactive API docs are at `http://127.0.0.1:8000/docs`
if you want to call endpoints directly.

**Using the UI:**
1. Set count / fraud % / seed at the top left, click **Generate bookings**.
2. Click any booking in the list — the agent investigates it live and the
   case file on the right fills in with a risk gauge, flags, explanation,
   and stamped verdict.
3. Click **Analyze all loaded** to batch-process everything and see an
   accuracy summary against the synthetic ground truth.

To stop the server: click into the terminal, press `Ctrl+C`.

---

## A note on free-tier rate limits

Each transaction can involve several tool calls, i.e. several API round
trips. The agent automatically retries with exponential backoff on
rate-limit (429) and overload (503) errors. On a free key, keep batches
small (10–30 transactions) and expect large batch runs to be slow or need
retries.

---

## Deploying a live demo

The app is a single FastAPI service that serves both the UI and the API, so
one deployment covers everything — no separate frontend hosting needed.

### Option A: Render (recommended, no Docker needed)

1. Push this project to GitHub (see the git section below).
2. Go to https://render.com → sign up (free, no card for the free tier) →
   **New +** → **Web Service** → connect your GitHub repo.
3. Render will detect `render.yaml` automatically and pre-fill the service
   config (Python env, build/start commands). If it doesn't auto-detect,
   set manually:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Under **Environment**, add your secrets (these are the `sync: false`
   entries in `render.yaml`, so Render will prompt for them):
   - `GROQ_API_KEY` (or `GEMINI_API_KEY` if using Gemini — also set
     `LLM_PROVIDER=gemini` to override the default)
5. Click **Create Web Service**. First deploy takes a few minutes. Render
   gives you a live URL like `https://fraud-detector.onrender.com`.

> Free-tier Render services spin down after inactivity and take ~30-60s to
> wake up on the next request — normal for a free demo, not a bug.

### Option B: Hugging Face Spaces (Docker)

1. Go to https://huggingface.co/new-space → choose a name → **Docker** as
   the SDK → **Create Space**.
2. At the very top of your `README.md`, add this metadata block (Spaces
   requires it to know how to build the Space):
   ```yaml
   ---
   title: Fraud Detector
   emoji: 🛂
   colorFrom: yellow
   colorTo: red
   sdk: docker
   app_port: 7860
   ---
   ```
3. Push this project (including the `Dockerfile`) to the Space's git repo:
   ```powershell
   git remote add space https://huggingface.co/spaces/yourname/fraud-detector
   git push space main
   ```
4. In the Space's **Settings → Repository secrets**, add `GROQ_API_KEY`
   (or `GEMINI_API_KEY` + `LLM_PROVIDER=gemini`).
5. The Space builds automatically from the `Dockerfile` and serves at
   `https://huggingface.co/spaces/yourname/fraud-detector`.

### Option C: Any other Docker host

The included `Dockerfile` is generic — it works on Railway, Fly.io, Google
Cloud Run, or any platform that builds from a `Dockerfile` and respects a
`$PORT` env var. Set `GROQ_API_KEY` (or `GEMINI_API_KEY` + `LLM_PROVIDER`)
as a secret/environment variable on whichever platform you choose.

---

## Pushing this to GitHub

You're working inside a `venv`, which is good — it's already excluded from
git via `.gitignore`, along with `.env` (your API key) and `__pycache__`.

### 1. Work from the project root
Run these commands from `D:\fraud_detector`, not from inside `venv\`. Being
*activated* into the venv (prompt shows `(venv)`) is fine — that's
different from being *inside* the `venv` folder.

### 2. Initialize git (skip if already done)
```powershell
git init
```

### 3. Check what will be committed
```powershell
git status
```
You should **not** see `venv/` or `.env` listed. If you do, confirm
`.gitignore` is saved in the project root, named exactly `.gitignore`.

### 4. Stage and commit
```powershell
git add .
git commit -m "Fraud detection agent with UI, Groq/Gemini support, deploy config"
```

### 5. Create a GitHub repo
Go to https://github.com/new, create an empty repo (don't initialize with
a README), and copy the URL it gives you.

### 6. Connect and push
```powershell
git branch -M main
git remote add origin https://github.com/yourname/fraud-detector.git
git push -u origin main
```

### 7. Future changes
```powershell
git add .
git commit -m "describe what changed"
git push
```

### If `git` isn't recognized
Install it from https://git-scm.com/download/win, restart VS Code's
terminal, then retry step 2.

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
