"""
Deterministic graders for all three task levels.

Key design principles:
  1. Proportional signal scoring: reward = 0.20 * (caught / total_gt_signals)
  2. Only ground-truth signals for that transaction count — not all 8 sub-factors
  3. Earlier pattern detection is rewarded more in medium
  4. Episode-end portfolio bonus for hard (F1, recall, FPR thresholds)
  5. Penalties are mutually exclusive — FP/FN replace base wrong-classification penalty
"""
from __future__ import annotations
from typing import Dict, Any, List

from models import FraudAction


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _correct(action: FraudAction, txn: Dict[str, Any]) -> bool:
    return action.classification == txn["_ground_truth"]


def _action_correct(action: FraudAction, txn: Dict[str, Any]) -> bool:
    return action.recommended_action == txn["_ground_truth_action"]


def _signal_score(action: FraudAction, txn: Dict[str, Any]) -> float:
    """
    Proportional signal credit.
    Only the ground-truth signals for THIS transaction are evaluated.

      gt_signals = [] (legitimate) → +0.20 for raising no spurious signals, 0.0 otherwise
      gt_signals = [a,b,c]        → 0.20 * (caught / 3)
    """
    gt: List[str] = txn.get("_ground_truth_signals", [])
    if not gt:
        return 0.20 if not action.triggered_signals else 0.0
    caught = len(set(action.triggered_signals) & set(gt))
    return round(0.20 * caught / len(gt), 4)


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 1 — EASY
# ══════════════════════════════════════════════════════════════════════════════

def grade_easy(action: FraudAction, txn: Dict[str, Any]) -> float:
    """
    Score a single easy-task transaction.

    Breakdown (max 1.0):
      Classification correct              +0.35
      Signal quality (proportional)       0.00–0.20
      Correct recommended action          +0.25
      Confidence calibration (≥ 0.6)      +0.10 / +0.05
      No false positive on legit          +0.10
    """
    score = 0.0
    gt = txn["_ground_truth"]
    correct = _correct(action, txn)

    if correct:
        score += 0.35

    # Proportional signal credit — only relevant signals for this txn
    score += _signal_score(action, txn)

    if _action_correct(action, txn):
        score += 0.25

    # Confidence calibration
    if correct and action.confidence >= 0.60:
        score += 0.10
    elif correct:
        score += 0.05

    # Bonus: agent correctly identifies legitimate without false-flagging
    if gt == "legitimate" and action.classification == "legitimate":
        score += 0.10

    return round(min(1.0, max(0.0, score)), 4)


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 2 — MEDIUM
# ══════════════════════════════════════════════════════════════════════════════

def grade_medium(
    action: FraudAction,
    txn: Dict[str, Any],
    step: int,
    total_steps: int,
) -> float:
    """
    Score one step of the pattern-recognition episode.

    Earlier detection of fraud/suspicious is rewarded more:
      catching the pattern at step 4 (of 8) > catching it at step 7.

    Breakdown (max 1.0):
      Classification correct              +0.30
      Proportional signal quality         0.00–0.20
      Correct recommended action          +0.20
      Early detection bonus               0.00–0.20
      No false positive on legit          +0.10
    """
    score = 0.0
    gt = txn["_ground_truth"]
    correct = _correct(action, txn)

    if correct:
        score += 0.30

    score += _signal_score(action, txn)

    if _action_correct(action, txn):
        score += 0.20

    # Early detection bonus — only meaningful when the txn is actually fraud/suspicious
    if gt in ("fraud", "suspicious") and correct:
        # step is 0-indexed; earlier steps earn more bonus (max 0.20 at step 0)
        early_bonus = 0.20 * max(0.0, (total_steps - 1 - step) / max(total_steps - 1, 1))
        score += early_bonus

    # No false positive on legitimates
    if gt == "legitimate" and action.classification == "legitimate":
        score += 0.10

    return round(min(1.0, max(0.0, score)), 4)


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 3 — HARD (per-step)
# ══════════════════════════════════════════════════════════════════════════════

def grade_hard_step(action: FraudAction, txn: Dict[str, Any]) -> float:
    """
    Per-step grade for the hard portfolio task.

    Breakdown (max 1.0):
      Classification correct              +0.35
      Proportional signal quality         0.00–0.20
      Correct recommended action          +0.25
      No false positive on legit          +0.10
      Confidence calibration              +0.10 / 0.00

    Penalties (replace base wrong-classification, not added on top):
      FP (fraud on legitimate)            -0.30
      FN (legitimate on fraud)            -0.40
      flag_for_review on legitimate       -0.15  (flag-everything exploit prevention)
    """
    score = 0.0
    gt = txn["_ground_truth"]
    correct = _correct(action, txn)

    # ── Classification (mutually exclusive penalties) ──────────────────────
    if correct:
        score += 0.35
    elif gt == "legitimate" and action.classification == "fraud":
        score -= 0.30   # FP replaces base -0.20
    elif gt == "fraud" and action.classification == "legitimate":
        score -= 0.40   # FN replaces base -0.20 (heaviest)
    else:
        score -= 0.20   # other misclassification

    # ── Flag-everything exploit prevention ────────────────────────────────
    if gt == "legitimate" and action.recommended_action == "flag_for_review":
        score -= 0.15

    # ── Signal quality ────────────────────────────────────────────────────
    score += _signal_score(action, txn)

    # ── Recommended action ────────────────────────────────────────────────
    if _action_correct(action, txn):
        score += 0.25

    # ── No false positive bonus ───────────────────────────────────────────
    if gt == "legitimate" and action.classification == "legitimate":
        score += 0.10

    # ── Confidence calibration ────────────────────────────────────────────
    if correct and action.confidence > 0.70:
        score += 0.10

    return round(min(1.0, max(0.0, score)), 4)


# ══════════════════════════════════════════════════════════════════════════════
#  TASK 3 — HARD (episode-level portfolio bonus)
# ══════════════════════════════════════════════════════════════════════════════

def grade_hard_portfolio_bonus(
    actions: List[FraudAction],
    transactions: List[Dict[str, Any]],
) -> float:
    """
    Episode-end portfolio bonus applied once at the final step.

    Evaluates the agent's portfolio-level performance:
      F1 > 0.90              → +0.30
      Fraud recall > 0.95    → +0.20
      FP rate < 0.05         → +0.20
      All three achieved     → +0.70 total (not additive beyond this)

    Returns the bonus (0.0 – 0.70) to be added to the final step reward.
    """
    if not actions:
        return 0.0

    # True/false positive and negative counts
    tp = sum(
        1 for a, t in zip(actions, transactions)
        if t["_ground_truth"] in ("fraud", "suspicious")
        and a.classification in ("fraud", "suspicious")
    )
    fp = sum(
        1 for a, t in zip(actions, transactions)
        if t["_ground_truth"] == "legitimate"
        and a.classification in ("fraud", "suspicious")
    )
    fn = sum(
        1 for a, t in zip(actions, transactions)
        if t["_ground_truth"] in ("fraud", "suspicious")
        and a.classification == "legitimate"
    )
    tn = sum(
        1 for a, t in zip(actions, transactions)
        if t["_ground_truth"] == "legitimate"
        and a.classification == "legitimate"
    )

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    total_legit = fp + tn
    fpr = fp / total_legit if total_legit > 0 else 0.0

    bonus = 0.0
    if f1 > 0.90:
        bonus += 0.30
    if recall > 0.95:
        bonus += 0.20
    if fpr < 0.05:
        bonus += 0.20

    return round(min(0.70, bonus), 4)
