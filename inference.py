"""
inference.py — Official submission inference script.

Mandatory environment variables:
  API_BASE_URL      The API endpoint for the LLM (default: https://router.huggingface.co/v1)
  MODEL_NAME        The model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN          Your Hugging Face / API key
  LOCAL_IMAGE_NAME  Local docker image name (if using from_docker_image)

Optional:
  ENV_URL           URL of the fraud detection environment server (default: http://localhost:8000)
  TASK_NAME         Specific task to run: easy | medium | hard (default: runs all 3)

STDOUT FORMAT (mandatory — do not change):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""
from __future__ import annotations

import os
import sys
import textwrap
import time
import json
from typing import List, Optional

from openai import OpenAI
from client import FraudDetectionEnv, StepResult
from models import FraudAction, FraudObservation

# ── Environment variables ─────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME   = os.getenv("MODEL_NAME")   or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL      = os.getenv("ENV_URL")      or "http://localhost:8000"
TASK_NAME    = os.getenv("TASK_NAME")    or None   # None = run all 3 tasks
BENCHMARK    = "fraud_detection"
IMAGE_NAME   = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")

SUCCESS_SCORE_THRESHOLD = 0.1   # normalised score in [0, 1]

# ── OpenAI client pointed at HuggingFace router ───────────────────────────────
client_ai = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
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

    Available signals:
    - location_mismatch, velocity_spike, amount_anomaly, unusual_time
    - account_takeover_pattern, card_testing_pattern, new_device
    - international_transaction, high_risk_merchant, chargeback_history
    - money_mule_pattern, synthetic_identity, sudden_large_transaction
    - no_prior_international_history, rapid_merchant_change

    Classification: "fraud"=decline immediately, "suspicious"=needs review, "legitimate"=allow
    Action: "allow" | "flag_for_review" | "decline" | "request_verification"

    Reasoning steps:
    1. Compare merchant_location vs card_holder_location
    2. Check velocity_last_hour (3+ elevated, 5+ suspicious)
    3. Check amount_vs_avg_ratio (2x notable, 4x+ anomalous)
    4. Read transaction_history timestamps for dormancy
    5. Check account_age_days and previous_fraud_flags
    6. Combine ALL weak signals — no single field is decisive
""").strip()


# ══════════════════════════════════════════════════════════════════════════════
#  Mandatory log functions — format must not change
# ══════════════════════════════════════════════════════════════════════════════

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Model interaction
# ══════════════════════════════════════════════════════════════════════════════

def build_user_message(obs: FraudObservation) -> str:
    history_str = "\n".join(
        f"  - Rs.{t['amount']:,.0f} at {t['merchant']} in {t['location']} ({t['timestamp'][:10]})"
        for t in obs.transaction_history
    ) or "  (no history)"

    feedback_str = "\n".join(
        f"  [{m}]" for m in obs.messages
    ) if obs.messages else "  (none)"

    return textwrap.dedent(f"""
        TRANSACTION:
          ID              : {obs.transaction_id}
          Amount          : Rs.{obs.amount:,.2f} {obs.currency}
          Merchant Type   : {obs.merchant_category}
          Merchant City   : {obs.merchant_location}
          Card Holder     : {obs.card_holder_location}
          Time of Day     : {obs.time_of_day}
          Day of Week     : {obs.day_of_week}

        ACCOUNT:
          Age (days)      : {obs.account_age_days}
          Fraud Flags     : {obs.previous_fraud_flags}
          Velocity (1hr)  : {obs.velocity_last_hour}
          Amount/30d Avg  : {obs.amount_vs_avg_ratio:.2f}x

        HISTORY (last 5):
        {history_str}

        ENV FEEDBACK:
        {feedback_str}
    """).strip()


def call_model(obs: FraudObservation, max_retries: int = 3) -> tuple[FraudAction, str]:
    """
    Returns (FraudAction, action_str_for_log).
    action_str format: classification|recommended_action
    """
    user_msg = build_user_message(obs)

    for attempt in range(max_retries):
        try:
            response = client_ai.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=300,
                stream=False,
            )
            raw = (response.choices[0].message.content or "").strip()

            # Strip markdown fences if present
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

            reasoning = str(data.get("reasoning", "No reasoning."))

            action = FraudAction(
                classification=classification,
                confidence=confidence,
                triggered_signals=signals,
                recommended_action=rec_action,
                reasoning=reasoning,
            )
            action_str = f"{classification}|{rec_action}"
            return action, action_str

        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            # Fallback
            action = FraudAction(
                classification="suspicious",
                confidence=0.5,
                triggered_signals=[],
                recommended_action="flag_for_review",
                reasoning=f"Parse failed: {exc}",
            )
            return action, "suspicious|flag_for_review"


# ══════════════════════════════════════════════════════════════════════════════
#  Episode runner
# ══════════════════════════════════════════════════════════════════════════════

def run_task(env_client: FraudDetectionEnv, task_level: str) -> float:
    """
    Run one episode. Emits [START] / [STEP] / [END] logs.
    Returns normalised score in [0.0, 1.0].
    [END] is always emitted, even on exception.
    """
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task_level, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env_client.reset(task_level)

        while not obs.done:
            steps_taken += 1
            error_str: Optional[str] = None

            try:
                action, action_str = call_model(obs)
                result: StepResult  = env_client.step(action)
                reward = result.reward if result.reward is not None else 0.0
                done   = result.done
                obs    = result.observation
            except Exception as exc:
                reward    = 0.0
                done      = True
                error_str = str(exc)
                action_str = "null"

            rewards.append(reward)
            log_step(
                step=steps_taken,
                action=action_str,
                reward=reward,
                done=done,
                error=error_str,
            )

            if done:
                break

        # Normalise: raw reward in [-1, 1] → score in [0, 1]
        raw_avg = sum(rewards) / len(rewards) if rewards else 0.0
        score   = round(min(max((raw_avg + 1.0) / 2.0, 0.0), 1.0), 3)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        score   = 0.0
        success = False

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return score


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    if not API_KEY:
        print("[DEBUG] HF_TOKEN / API_KEY not set.", flush=True)
        sys.exit(1)

    # Determine which tasks to run
    if TASK_NAME and TASK_NAME in ("easy", "medium", "hard"):
        tasks = [TASK_NAME]
    else:
        tasks = ["easy", "medium", "hard"]

    with FraudDetectionEnv(base_url=ENV_URL).sync() as env_client:
        # Verify server is reachable
        try:
            env_client.health()
        except Exception as exc:
            print(f"[DEBUG] Cannot reach environment at {ENV_URL}: {exc}", flush=True)
            sys.exit(1)

        for task_level in tasks:
            run_task(env_client, task_level)


if __name__ == "__main__":
    main()
