"""
Baseline inference script.

Runs a GPT-4o-mini model against all 3 task levels using the fraud detection
environment and produces a reproducible summary score table.

Usage:
    export OPENAI_API_KEY=sk-...
    export ENV_URL=http://localhost:8000   # optional, defaults to localhost
    python baseline_inference.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Optional

from openai import OpenAI
from client import FraudDetectionEnv, StepResult
from models import FraudAction, FraudObservation

# ── Configuration ─────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")
MODEL = "gpt-4o-mini"

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
- unusual_time               (transaction at 2-4am local time)
- account_takeover_pattern   (password change + new device + new location)
- card_testing_pattern       (many small transactions at different merchants)
- new_device                 (transaction from unrecognised device)
- international_transaction  (transaction outside home country)
- high_risk_merchant         (crypto, gambling, forex merchant)
- chargeback_history         (account has prior chargeback disputes)
- money_mule_pattern         (deposit + immediate withdrawal cycle)
- synthetic_identity         (new account, rapidly escalating amounts)
- sudden_large_transaction   (first large transaction after dormant period)
- no_prior_international_history (first ever international transaction)
- rapid_merchant_change      (many different merchants in short window)

Classification rules:
- "fraud":       clear fraudulent activity, block immediately
- "suspicious":  possible fraud, needs review or verification
- "legitimate":  normal transaction, allow

Recommended action rules:
- "allow":                 safe transaction, proceed
- "flag_for_review":       suspicious, human review needed (do NOT use on clearly legitimate transactions)
- "decline":               fraud confirmed, reject this specific transaction
- "request_verification":  verify customer identity before allowing

Think carefully about:
1. Is the merchant location different from the card holder location?
2. How many transactions in the last hour (velocity)?
3. Is the amount much higher than the 30-day average (amount_vs_avg_ratio)?
4. Does the transaction history show suspicious patterns?
5. Are there previous fraud flags on the account?
"""

# ──────────────────────────────────────────────────────────────────────────────

def build_user_message(obs: FraudObservation) -> str:
    """Convert a FraudObservation into a human-readable prompt."""
    history_str = "\n".join(
        f"  - ₹{t['amount']:,.0f} at {t['merchant']} in {t['location']} ({t['timestamp'][:10]})"
        for t in obs.transaction_history
    )
    messages_str = "\n".join(f"  [{m}]" for m in obs.messages) if obs.messages else "  (none)"

    return f"""
TRANSACTION TO ANALYSE:
  Transaction ID : {obs.transaction_id}
  Amount         : ₹{obs.amount:,.2f} {obs.currency}
  Merchant Type  : {obs.merchant_category}
  Merchant City  : {obs.merchant_location}
  Card Holder    : {obs.card_holder_location}
  Time of Day    : {obs.time_of_day}
  Day of Week    : {obs.day_of_week}

ACCOUNT CONTEXT:
  Account Age    : {obs.account_age_days} days
  Prev Fraud Flags: {obs.previous_fraud_flags}
  Velocity (1h)  : {obs.velocity_last_hour} transactions
  Amount/Avg30d  : {obs.amount_vs_avg_ratio:.2f}x normal

RECENT TRANSACTION HISTORY (last 5):
{history_str}

ENVIRONMENT FEEDBACK:
{messages_str}
"""


def call_ai(client: OpenAI, obs: FraudObservation, max_retries: int = 3) -> FraudAction:
    """Call the AI model and parse its JSON response into a FraudAction."""
    user_msg = build_user_message(obs)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.strip()},
                    {"role": "user", "content": user_msg.strip()},
                ],
                temperature=0.0,  # deterministic
                seed=42,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            data = json.loads(raw)

            # Sanitise fields
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

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            # Fallback action on repeated failures
            print(f"  WARNING: AI parse failed ({exc}), using fallback action.")
            return FraudAction(
                classification="suspicious",
                confidence=0.5,
                triggered_signals=[],
                recommended_action="flag_for_review",
                reasoning="Could not parse AI response; defaulting to suspicious.",
            )


def run_task(
    client_ai: OpenAI,
    env: "SyncWrapper",
    task_level: str,
    task_name: str,
) -> float:
    """Run one full episode and return the final average score."""
    bar = "═" * 55
    print(f"\n{bar}")
    print(f"TASK ({task_level.upper()}): {task_name}")
    print(bar)

    obs = env.reset(task_level)
    step_rewards: list[float] = []
    step_num = 0

    while not obs.done:
        step_num += 1
        action = call_ai(client_ai, obs)

        result: StepResult = env.step(action)
        reward = result.reward or 0.0
        step_rewards.append(reward)
        obs = result.observation

        # Pretty print
        amount_str = f"₹{obs.amount:,.0f}" if step_num == 1 else f"₹{obs.amount:,.0f}"
        print(
            f"\nTransaction {step_num}: {amount_str} | "
            f"{obs.merchant_location} | {obs.time_of_day}"
        )
        print(
            f"  AI Decision : {action.classification.upper()} "
            f"(confidence: {action.confidence:.2f})"
        )
        if action.triggered_signals:
            print(f"  Signals     : {', '.join(action.triggered_signals)}")
        else:
            print("  Signals     : (none)")
        print(f"  Action      : {action.recommended_action}")
        print(f"  Reasoning   : {action.reasoning}")
        print(f"  Reward      : {reward:+.2f}")

        if obs.messages:
            for msg in obs.messages:
                print(f"  Env Feedback: {msg}")

    final_score = (
        sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
    )
    # Normalise to [0, 1] for display (rewards are in [-1, 1])
    normalised = (final_score + 1.0) / 2.0
    print(f"\nFinal Score (normalised): {normalised:.4f}")
    return normalised


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    client_ai = OpenAI(api_key=OPENAI_API_KEY)

    tasks = [
        ("easy",   "Single Transaction Screening"),
        ("medium", "Pattern Recognition"),
        ("hard",   "Mixed Portfolio Review"),
    ]

    scores: dict[str, float] = {}

    with FraudDetectionEnv(base_url=ENV_URL).sync() as env:
        # Verify server is healthy
        try:
            h = env.health()
            print(f"Server status: {h}")
        except Exception as exc:
            print(f"ERROR: Cannot connect to environment at {ENV_URL}: {exc}")
            sys.exit(1)

        for level, name in tasks:
            scores[level] = run_task(client_ai, env, level, name)

    # ── Summary table ─────────────────────────────────────────────────────────
    bar = "═" * 55
    print(f"\n\n{bar}")
    print("BASELINE RESULTS SUMMARY")
    print(bar)
    print(f"{'Task':<30} {'Score':>10}")
    print("-" * 42)
    print(f"{'Task 1 (Easy)':<30} {scores.get('easy', 0.0):>10.4f}")
    print(f"{'Task 2 (Medium)':<30} {scores.get('medium', 0.0):>10.4f}")
    print(f"{'Task 3 (Hard)':<30} {scores.get('hard', 0.0):>10.4f}")
    print(bar)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"{'Overall Average':<30} {avg:>10.4f}")
    print(bar)


if __name__ == "__main__":
    main()
