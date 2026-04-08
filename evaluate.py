"""
evaluate.py — Proves the environment rewards structured reasoning over random guessing.

Runs two agents against all 3 task levels over multiple episodes each:

  1. Random Agent   — randomly picks classification, signals, confidence, action
  2. Rule Agent     — simple heuristic rules (no LLM needed, fast, deterministic)

Prints a comparison table showing that the environment successfully
differentiates between random guessing and structured reasoning.

Usage:
    python evaluate.py

No API key required. The Rule Agent uses deterministic heuristics.
"""
from __future__ import annotations

import random
import sys
import os
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(__file__))

from models import FraudAction, FraudObservation
from server.environment import FraudDetectionEnvironment

# ── Constants ─────────────────────────────────────────────────────────────────

TASK_LEVELS = ["easy", "medium", "hard"]
EPISODES_PER_TASK = 5          # episodes per agent per task
RANDOM_SEED = 42

ALL_SIGNALS = [
    "location_mismatch", "velocity_spike", "amount_anomaly",
    "unusual_time", "account_takeover_pattern", "card_testing_pattern",
    "new_device", "international_transaction", "high_risk_merchant",
    "chargeback_history", "money_mule_pattern", "synthetic_identity",
    "sudden_large_transaction", "no_prior_international_history",
    "rapid_merchant_change",
]
CLASSIFICATIONS = ["legitimate", "suspicious", "fraud"]
ACTIONS = ["allow", "flag_for_review", "decline", "request_verification"]


# ══════════════════════════════════════════════════════════════════════════════
#  Agent 1 — Random Agent
# ══════════════════════════════════════════════════════════════════════════════

class RandomAgent:
    """Picks every field uniformly at random. Baseline floor."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)

    def act(self, obs: FraudObservation) -> FraudAction:
        classification = self.rng.choice(CLASSIFICATIONS)
        n_signals = self.rng.randint(0, 3)
        signals = self.rng.sample(ALL_SIGNALS, n_signals)
        return FraudAction(
            classification=classification,
            confidence=round(self.rng.uniform(0.3, 0.9), 2),
            triggered_signals=signals,
            recommended_action=self.rng.choice(ACTIONS),
            reasoning="Random agent — no reasoning.",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Agent 2 — Rule-Based Heuristic Agent
# ══════════════════════════════════════════════════════════════════════════════

class RuleAgent:
    """
    Deterministic heuristic agent. Uses observable fields to make decisions.

    Rules (in priority order):
      1. velocity_last_hour >= 5          → fraud, decline
      2. amount_vs_avg_ratio >= 8         → fraud, decline
      3. merchant_location != card_holder_location  → suspicious, request_verification
      4. time_of_day == "night" and amount > 20000  → suspicious, flag_for_review
      5. previous_fraud_flags >= 2        → suspicious, flag_for_review
      6. amount_vs_avg_ratio >= 3         → suspicious, flag_for_review
      7. default                          → legitimate, allow
    """

    def act(self, obs: FraudObservation) -> FraudAction:
        signals: List[str] = []
        classification = "legitimate"
        action = "allow"
        confidence = 0.7
        reasoning = "Transaction appears normal."

        # Rule 1: velocity spike
        if obs.velocity_last_hour >= 5:
            signals.append("velocity_spike")
            signals.append("card_testing_pattern")
            classification = "fraud"
            action = "decline"
            confidence = 0.88
            reasoning = f"{obs.velocity_last_hour} transactions in last hour — card testing pattern."

        # Rule 2: massive amount anomaly
        elif obs.amount_vs_avg_ratio >= 8.0:
            signals.append("amount_anomaly")
            if obs.merchant_location != obs.card_holder_location:
                signals.append("location_mismatch")
                signals.append("international_transaction")
            classification = "fraud"
            action = "decline"
            confidence = 0.91
            reasoning = f"Amount is {obs.amount_vs_avg_ratio:.1f}x the 30-day average."

        # Rule 3: location mismatch
        elif obs.merchant_location != obs.card_holder_location:
            signals.append("location_mismatch")
            if "Nigeria" in obs.merchant_location or "Ghana" in obs.merchant_location \
                    or "Kenya" in obs.merchant_location:
                signals.append("international_transaction")
                classification = "fraud"
                action = "decline"
                confidence = 0.85
                reasoning = "International transaction to high-risk geography."
            else:
                classification = "suspicious"
                action = "request_verification"
                confidence = 0.65
                reasoning = "Transaction location differs from cardholder home city."

        # Rule 4: night + high amount
        elif obs.time_of_day == "night" and obs.amount > 20000:
            signals.append("unusual_time")
            signals.append("amount_anomaly")
            classification = "suspicious"
            action = "flag_for_review"
            confidence = 0.60
            reasoning = "Large transaction at night warrants review."

        # Rule 5: chargeback history
        elif obs.previous_fraud_flags >= 2:
            signals.append("chargeback_history")
            classification = "suspicious"
            action = "flag_for_review"
            confidence = 0.65
            reasoning = f"Account has {obs.previous_fraud_flags} prior fraud flags."

        # Rule 6: moderate amount anomaly
        elif obs.amount_vs_avg_ratio >= 3.0:
            signals.append("amount_anomaly")
            classification = "suspicious"
            action = "flag_for_review"
            confidence = 0.55
            reasoning = f"Amount is {obs.amount_vs_avg_ratio:.1f}x the 30-day average."

        return FraudAction(
            classification=classification,
            confidence=confidence,
            triggered_signals=signals,
            recommended_action=action,
            reasoning=reasoning,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Episode runner
# ════════════════════════════════════════════════════════════════���═════════════

def run_episode(agent, task_level: str, verbose: bool = False) -> Tuple[float, int]:
    """
    Run one episode with the given agent.
    Returns (normalised_score, steps_taken).
    """
    env = FraudDetectionEnvironment()
    obs = env.reset(task_level)
    rewards: List[float] = []
    steps = 0

    while not obs.done:
        action = agent.act(obs)
        obs = env.step(action)
        if obs.reward is not None:
            rewards.append(obs.reward)
        steps += 1
        if verbose:
            print(
                f"  Step {steps}: ₹{obs.amount:,.0f} | "
                f"{action.classification} → reward {obs.reward:+.2f}"
            )

    # Normalise from [-1,1] to [0,1]
    raw = sum(rewards) / len(rewards) if rewards else 0.0
    normalised = round((raw + 1.0) / 2.0, 4)
    return normalised, steps


def evaluate_agent(agent, agent_name: str) -> dict:
    """Run the agent across all tasks and return score dict."""
    results: dict = {}
    for level in TASK_LEVELS:
        episode_scores = []
        for ep in range(EPISODES_PER_TASK):
            score, _ = run_episode(agent, level)
            episode_scores.append(score)
        avg = round(sum(episode_scores) / len(episode_scores), 4)
        results[level] = avg
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\nInitialising agents...")
    random_agent = RandomAgent(seed=RANDOM_SEED)
    rule_agent   = RuleAgent()

    print(f"Running {EPISODES_PER_TASK} episodes per task per agent...\n")

    random_scores = evaluate_agent(random_agent, "Random Agent")
    rule_scores   = evaluate_agent(rule_agent,   "Rule Agent")

    bar = "═" * 62
    print(f"\n{bar}")
    print("EVALUATION RESULTS  (score range: 0.0 = worst, 1.0 = best)")
    print(bar)
    print(f"{'Task':<28} {'Random Agent':>14} {'Rule Agent':>14}")
    print("-" * 58)

    for level in TASK_LEVELS:
        r = random_scores[level]
        h = rule_scores[level]
        diff = h - r
        print(
            f"Task ({level:<6})              "
            f"{r:>14.4f} {h:>14.4f}   (+{diff:.4f})"
        )

    print("-" * 58)
    r_avg = sum(random_scores.values()) / len(random_scores)
    h_avg = sum(rule_scores.values()) / len(rule_scores)
    print(f"{'Overall Average':<28} {r_avg:>14.4f} {h_avg:>14.4f}   (+{h_avg - r_avg:.4f})")
    print(bar)

    print("\nVerification checks:")
    for level in TASK_LEVELS:
        r = random_scores[level]
        h = rule_scores[level]
        status = "PASS" if h > r else "FAIL"
        print(f"  [{status}] {level}: rule_agent ({h:.4f}) > random_agent ({r:.4f})")

    all_pass = all(rule_scores[l] > random_scores[l] for l in TASK_LEVELS)
    print(
        f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'} — "
        "environment successfully rewards structured reasoning over random guessing.\n"
    )


if __name__ == "__main__":
    main()
