"""
inference.py — Official submission inference script.

Environment variables required:
  API_BASE_URL  — HuggingFace inference endpoint base URL
  MODEL_NAME    — Model identifier (e.g. "meta-llama/Llama-3.1-8B-Instruct")
  HF_TOKEN      — HuggingFace API key

Emits structured stdout logs in [START] / [STEP] / [END] format
as required by the hackathon evaluation system.

Usage:
  export API_BASE_URL=https://api-inference.huggingface.co/v1
  export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
  export HF_TOKEN=hf_...
  python inference.py
"""
from __future__ import annotations

import json
import os
import sys
import time

from openai import OpenAI
from client import FraudDetectionEnv, StepResult
from models import FraudAction, FraudObservation

# ── Environment variables ─────────────────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")
ENV_URL      = os.environ.get("ENV_URL",      "http://localhost:8000")

# ── OpenAI client pointed at HuggingFace inference endpoint ──────────────────
client_ai = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a financial fraud analyst AI for an Indian bank. You will receive transaction
details and must classify each transaction and identify fraud signals.

You must respond ONLY with this exact JSON format (no markdown, no extra text):
{
  "classification": "legitimate" | "suspicious" | "fraud",
  "confidence": 0.0 to 1.0,
  "triggered_signals": ["signal1", "signal2"],
  "recommended_action": "allow" | "flag_for_review" | "decline" | "request_verification",
  "reasoning": "one sentence explanation"
}

Available signals to detect:
- location_mismatch          (card holder location != transaction location)
- velocity_spike             (too many transactions in short time)
- amount_anomaly             (amount >> normal for this account)
- unusual_time               (transaction at unusual hours)
- account_takeover_pattern   (password change + new device + new location combo)
- card_testing_pattern       (many small transactions at different merchants)
- new_device                 (transaction from unrecognised device)
- international_transaction  (transaction outside home country)
- high_risk_merchant         (crypto, gambling, forex merchant)
- chargeback_history         (account has prior chargeback disputes)
- money_mule_pattern         (deposit + immediate withdrawal cycle)
- synthetic_identity         (new account with rapidly escalating amounts)
- sudden_large_transaction   (first large transaction after dormant period)
- no_prior_international_history (first ever international transaction)
- rapid_merchant_change      (many different merchants in short window)

Classification rules:
- "fraud":       clear fraudulent activity, decline immediately
- "suspicious":  possible fraud, needs review or verification
- "legitimate":  normal transaction, allow

Recommended action rules:
- "allow":                 safe transaction, proceed
- "flag_for_review":       suspicious, human review needed (NOT for clearly legitimate)
- "decline":               fraud confirmed, reject this specific transaction
- "request_verification":  verify customer identity before allowing

Key reasoning steps:
1. Compare merchant_location vs card_holder_location — same state or foreign?
2. Check velocity_last_hour — 3+ is elevated, 5+ is suspicious
3. Check amount_vs_avg_ratio — 2x is notable, 4x+ is anomalous
4. Read transaction_history timestamps — is the account dormant?
5. Look at account_age_days — new accounts with large txns = risk
6. Check previous_fraud_flags — even 1 flag + high amount = suspicious
7. Combine ALL weak signals — no single field is decisive
"""


def build_user_message(obs: FraudObservation) -> str:
    history_str = "\n".join(
        f"  - Rs.{t['amount']:,.0f} at {t['merchant']} in {t['location']} ({t['timestamp'][:10]})"
        for t in obs.transaction_history
    ) or "  (no history)"

    feedback_str = "\n".join(
        f"  [{m}]" for m in obs.messages
    ) if obs.messages else "  (none)"

    return f"""
TRANSACTION TO ANALYSE:
  Transaction ID  : {obs.transaction_id}
  Amount          : Rs.{obs.amount:,.2f} {obs.currency}
  Merchant Type   : {obs.merchant_category}
  Merchant City   : {obs.merchant_location}
  Card Holder     : {obs.card_holder_location}
  Time of Day     : {obs.time_of_day}
  Day of Week     : {obs.day_of_week}

ACCOUNT CONTEXT:
  Account Age     : {obs.account_age_days} days
  Fraud Flags     : {obs.previous_fraud_flags}
  Velocity (1hr)  : {obs.velocity_last_hour} transactions
  Amount/30d Avg  : {obs.amount_vs_avg_ratio:.2f}x

RECENT TRANSACTION HISTORY (last 5):
{history_str}

ENVIRONMENT FEEDBACK FROM PREVIOUS STEP:
{feedback_str}
"""


def call_model(obs: FraudObservation, max_retries: int = 3) -> FraudAction:
    """Call the LLM and parse response into a FraudAction."""
    user_msg = build_user_message(obs)

    for attempt in range(max_retries):
        try:
            response = client_ai.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.strip()},
                    {"role": "user",   "content": user_msg.strip()},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            raw = response.choices[0].message.content or ""

            # Strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            data = json.loads(raw)

            classification = data.get("classification", "suspicious")
            if classification not in ("legitimate", "suspicious", "fraud"):
                classification = "suspicious"

            rec_action = data.get("recommended_action", "flag_for_review")
            if rec_action not in ("allow", "flag_for_review", "decline", "request_verification"):
                rec_action = "flag_for_review"

            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            signals = data.get("triggered_signals", [])
            if not isinstance(signals, list):
                signals = []

            reasoning = str(data.get("reasoning", "No reasoning provided."))

            return FraudAction(
                classification=classification,
                confidence=confidence,
                triggered_signals=signals,
                recommended_action=rec_action,
                reasoning=reasoning,
            )

        except (json.JSONDecodeError, KeyError, ValueError, Exception) as exc:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            # Fallback on repeated failure
            return FraudAction(
                classification="suspicious",
                confidence=0.5,
                triggered_signals=[],
                recommended_action="flag_for_review",
                reasoning=f"Model parse failed: {exc}",
            )


def run_task(env: "SyncWrapper", task_level: str, episode_num: int) -> float:
    """
    Run one episode and emit [START] / [STEP] / [END] logs.
    Returns normalised score in [0.0, 1.0].
    """
    obs = env.reset(task_level)

    # ── [START] log ───────────────────────────────────────────────────────────
    print(json.dumps({
        "log_type": "START",
        "task":     task_level,
        "episode":  episode_num,
        "model":    MODEL_NAME,
    }), flush=True)

    step_num   = 0
    rewards: list[float] = []

    while not obs.done:
        step_num += 1
        action = call_model(obs)
        result: StepResult = env.step(action)
        reward = result.reward if result.reward is not None else 0.0
        rewards.append(reward)
        obs = result.observation

        # ── [STEP] log ────────────────────────────────────────────────────────
        print(json.dumps({
            "log_type":       "STEP",
            "task":           task_level,
            "episode":        episode_num,
            "step":           step_num,
            "transaction_id": obs.transaction_id,
            "amount":         obs.amount,
            "merchant_location": obs.merchant_location,
            "time_of_day":    obs.time_of_day,
            "classification": action.classification,
            "confidence":     action.confidence,
            "triggered_signals": action.triggered_signals,
            "recommended_action": action.recommended_action,
            "reasoning":      action.reasoning,
            "reward":         reward,
        }), flush=True)

    raw_avg   = sum(rewards) / len(rewards) if rewards else 0.0
    score     = round((raw_avg + 1.0) / 2.0, 4)   # normalise [-1,1] → [0,1]

    # ── [END] log ─────────────────────────────────────────────────────────────
    print(json.dumps({
        "log_type":    "END",
        "task":        task_level,
        "episode":     episode_num,
        "total_steps": step_num,
        "raw_reward":  round(raw_avg, 4),
        "score":       score,
    }), flush=True)

    return score


def main() -> None:
    if not HF_TOKEN:
        print(json.dumps({
            "log_type": "ERROR",
            "message":  "HF_TOKEN environment variable not set.",
        }), flush=True)
        sys.exit(1)

    tasks = [
        ("easy",   "Single Transaction Screening"),
        ("medium", "Pattern Recognition"),
        ("hard",   "Mixed Portfolio Review"),
    ]

    all_scores: dict[str, float] = {}

    with FraudDetectionEnv(base_url=ENV_URL).sync() as env:
        # Health check
        try:
            h = env.health()
            print(json.dumps({"log_type": "INFO", "message": f"Server healthy: {h}"}),
                  flush=True)
        except Exception as exc:
            print(json.dumps({"log_type": "ERROR", "message": f"Cannot reach env: {exc}"}),
                  flush=True)
            sys.exit(1)

        for ep_idx, (level, name) in enumerate(tasks, start=1):
            score = run_task(env, level, episode_num=ep_idx)
            all_scores[level] = score

    # ── Final summary ─────────────────────────────────────────────────────────
    print(json.dumps({
        "log_type": "SUMMARY",
        "scores":   all_scores,
        "average":  round(sum(all_scores.values()) / len(all_scores), 4),
    }), flush=True)


if __name__ == "__main__":
    main()
