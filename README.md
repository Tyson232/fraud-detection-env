---
title: Fraud Detection Environment
emoji: 💳
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
---

# Financial Fraud Detection Environment

A production-ready, OpenEnv-compliant reinforcement learning environment for
training and evaluating AI agents on financial fraud detection using realistic
Indian banking transaction data.

---

## Overview & Motivation

Financial fraud costs the Indian banking sector over ₹30,000 crore annually.
Traditional rule-based systems struggle to adapt to evolving fraud patterns and
generate high false-positive rates — blocking innocent customers and eroding
trust. A reasoning AI agent that understands context, remembers transaction
history, and calibrates its confidence is significantly better suited for this task.

**Why reinforcement learning?**
- Fraud patterns evolve; RL agents learn to adapt without rule rewrites
- The cost of errors is asymmetric: missing fraud is worse than over-flagging,
  but blocking innocent customers destroys trust — RL can encode this asymmetry
  directly in the reward function
- Sequential decision-making over transaction history mirrors real analyst workflows

**What makes this environment unique:**
- Rich Indian financial context (₹, UPI, Indian cities, realistic merchants)
- Asymmetric reward design reflecting real operational cost trade-offs
- Confidence calibration reward punishing overconfident wrong predictions
- Three difficulty levels from single-signal to portfolio-level reasoning
- Deterministic graders with per-step reward signals (not just episode-end)

---

## Environment Description

The agent acts as an AI fraud analyst reviewing bank transactions one at a time.
At each step it receives full transaction context including merchant details,
geographic data, velocity information, and the last 5 transactions as history.
The agent must classify the transaction, identify the active fraud signals,
recommend an action, and state its confidence and reasoning.

### What the agent observes
Full transaction context: amount, merchant, location, time, account history,
velocity metrics, and natural-language feedback from the previous step.

### What actions it can take
Classify the transaction, name the fraud signals it detected, choose a
recommended action (allow / flag / block / verify), and provide a confidence
score and one-sentence reasoning.

### How rewards are calculated
A shaped reward signal is emitted at **every step** (not just episode end),
combining classification accuracy, signal quality, action correctness, and
confidence calibration. Penalties are asymmetric — blocking an innocent customer
carries a heavier penalty than over-flagging, and missing real fraud carries the
heaviest penalty of all.

---

## Action Space

| Field | Type | Valid Values |
|---|---|---|
| `classification` | enum | `legitimate`, `suspicious`, `fraud` |
| `confidence` | float | 0.0 – 1.0 |
| `triggered_signals` | list[str] | See signal vocabulary below |
| `recommended_action` | enum | `allow`, `flag_for_review`, `block`, `request_verification` |
| `reasoning` | str | One sentence |

**Signal vocabulary:**

| Signal | Description |
|---|---|
| `location_mismatch` | Card holder city ≠ merchant city |
| `velocity_spike` | Too many transactions in short window |
| `amount_anomaly` | Amount >> 30-day average |
| `unusual_time` | Transaction between 2–4 am |
| `account_takeover_pattern` | Password change + new device + new location combo |
| `card_testing_pattern` | Many small amounts at different merchants |
| `new_device` | Transaction from unrecognised device |
| `international_transaction` | Merchant outside India |
| `high_risk_merchant` | Crypto, gambling, forex platform |
| `chargeback_history` | Account has prior dispute history |
| `money_mule_pattern` | Deposit → immediate withdrawal cycle |
| `synthetic_identity` | New account escalating amounts rapidly |
| `sudden_large_transaction` | First large tx after dormant period |
| `no_prior_international_history` | First ever international transaction |
| `rapid_merchant_change` | Many different merchants in one hour |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `transaction_id` | str | Unique transaction identifier |
| `amount` | float | Amount in INR (₹) |
| `currency` | str | Always `INR` |
| `merchant_category` | str | e.g. `grocery`, `electronics`, `wire_transfer` |
| `merchant_location` | str | City/country where merchant is located |
| `card_holder_location` | str | Registered home city of card holder |
| `time_of_day` | str | `morning`, `afternoon`, `evening`, or `night` |
| `day_of_week` | str | Full day name |
| `transaction_history` | list | Last 5 transactions: amount, merchant, location, timestamp |
| `account_age_days` | int | Days since account was opened |
| `previous_fraud_flags` | int | Number of prior fraud disputes |
| `velocity_last_hour` | int | Transactions in the past 60 minutes |
| `amount_vs_avg_ratio` | float | Current amount ÷ 30-day average |
| `messages` | list[str] | Feedback from environment about previous step |
| `done` | bool | `True` when episode has ended |
| `reward` | float \| null | Reward for the most recent action |

---

## Reward Function

All rewards are clipped to **[-1.0, 1.0]**.

| Component | Formula | Notes |
|---|---|---|
| Classification correct | +0.35 | Primary signal |
| Classification wrong | -0.20 | — |
| Signal quality | +0.20 × (correct ÷ total) | Partial credit |
| Correct recommended action | +0.25 | — |
| False positive (block innocent) | -0.30 | Blocking innocent customer |
| Block action on legitimate | -0.20 | Additional penalty |
| False negative (miss fraud) | **-0.40** | Heaviest penalty |
| Confidence good (correct + conf > 0.7) | +0.10 | Calibration reward |
| Confidence bad (wrong + conf > 0.8) | -0.10 | Overconfident penalty |

**Why asymmetric?**
Missing fraud allows financial loss and damages the bank.  
Blocking an innocent customer destroys trust and incurs regulatory risk.  
Both are bad — but at different rates. The reward function encodes this directly.

---

## Task Descriptions

### Task 1 — Easy: Single Transaction Screening

**Objective:** Classify individual transactions with obvious fraud signals.  
**Episode length:** 5 transactions  
**Mix:** ~60% fraud, ~40% legitimate  
**Difficulty:** Signals are individually strong — location mismatch, 10x amount, 3am time.
An agent just checking the top 2–3 features should score well.  
**Expected baseline score (GPT-4o-mini):** 0.70  
**Expected random agent score:** ~0.20

### Task 2 — Medium: Pattern Recognition

**Objective:** Identify fraud patterns across a sequence of 8 transactions.  
**Episode length:** 8 transactions  
**Mix:** Account takeover, friendly fraud, sleeper fraud, legitimates  
**Difficulty:** Each individual signal appears benign. Account takeover requires
recognising that a password change + new device + different city = compromise.
Requires sequential reasoning and memory across steps.  
**Expected baseline score:** 0.50  
**Expected random agent score:** ~0.15

### Task 3 — Hard: Mixed Portfolio Review

**Objective:** Review 15 transactions, optimise precision and recall simultaneously.  
**Episode length:** 15 transactions (6 fraud, 4 suspicious, 5 legitimate)  
**Mix:** Money mules, synthetic identities, first-party fraud + obvious legitimates as traps  
**Difficulty:** Sophisticated fraud patterns require multi-step reasoning. Legitimate
transactions are designed to tempt over-flagging. Graded on F1 score, with heavy
per-false-positive and per-false-negative penalties.  
**Expected baseline score:** 0.35  
**Expected random agent score:** ~0.10

---

## Baseline Scores

| Task | Difficulty | Random Agent | GPT-4o-mini | Perfect Agent |
|---|---|---|---|---|
| Single Transaction Screening | Easy | ~0.20 | ~0.70 | 1.0 |
| Pattern Recognition | Medium | ~0.15 | ~0.50 | 1.0 |
| Mixed Portfolio Review | Hard | ~0.10 | ~0.35 | 1.0 |

---

## Setup Instructions

### 1. Clone and install

```bash
git clone <repo-url>
cd fraud_detection_env
pip install -r requirements.txt
```

### 2. Run the server locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

The server exposes:
- `POST /reset` — start a fresh episode
- `POST /step` — submit one action
- `GET /state` — current episode metadata
- `GET /health` — health check
- `GET /web` — live browser dashboard
- `WS /ws` — persistent WebSocket session

### 3. Run the baseline script

```bash
export OPENAI_API_KEY=sk-...
export ENV_URL=http://localhost:8000

python baseline_inference.py
```

### 4. Quick API test

```bash
# Reset with easy task
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_level": "easy"}'

# Submit an action
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{
    "classification": "fraud",
    "confidence": 0.9,
    "triggered_signals": ["location_mismatch", "amount_anomaly"],
    "recommended_action": "block",
    "reasoning": "Transaction from foreign country at 3am with 10x normal amount."
  }'
```

---

## Docker Instructions

```bash
# Build from project root
cd fraud_detection_env
docker build -f server/Dockerfile -t fraud-detection-env:latest .

# Run
docker run -p 8000:8000 \
  -e WORKERS=4 \
  -e MAX_CONCURRENT_ENVS=100 \
  fraud-detection-env:latest
```

The container exposes port 8000. Health check is built-in (see Dockerfile).

---

## HF Spaces Deployment

1. Create a new HF Space with Docker SDK
2. Set the Dockerfile path to `server/Dockerfile`
3. Push the repository
4. The Space will auto-build and expose the environment on the public URL

```yaml
# In Space settings:
app_port: 8000
sdk: docker
```

Then set `ENV_URL=https://your-space.hf.space` when running `baseline_inference.py`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server port |
| `WORKERS` | `4` | Uvicorn worker processes |
| `MAX_CONCURRENT_ENVS` | `100` | Max simultaneous WebSocket sessions |
| `OPENAI_API_KEY` | *(required)* | OpenAI API key for baseline script |
| `ENV_URL` | `http://localhost:8000` | Environment URL for baseline script |

---

## WebSocket Usage

Each WebSocket connection gets its own isolated environment instance.

```python
import websockets, json, asyncio

async def run():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        # Reset
        await ws.send(json.dumps({"type": "reset", "task_level": "easy"}))
        obs = json.loads(await ws.recv())

        # Step
        action = {
            "classification": "fraud",
            "confidence": 0.88,
            "triggered_signals": ["location_mismatch"],
            "recommended_action": "block",
            "reasoning": "Geographic mismatch detected."
        }
        await ws.send(json.dumps({"type": "step", "action": action}))
        result = json.loads(await ws.recv())

asyncio.run(run())
```

---

## Licence

MIT — see `pyproject.toml`.
