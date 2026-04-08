"""
FraudDetectionEnvironment — fully reactive, single-account narrative model.

Key changes vs v1:
  - Sequences are AccountSequence objects; next(prev_action) drives reactivity
  - For hard: 3 sequences interleaved round-robin (one account per sequence)
  - FP/FN penalties are mutually exclusive (replace base wrong-classification)
  - flag_for_review on legitimate → -0.15 (blocks "flag everything" exploit)
  - Portfolio bonus applied at episode end for hard task
  - No state leak between episodes
"""
from __future__ import annotations

import uuid
import sys
import os
from typing import List, Optional, Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import FraudAction, FraudObservation, FraudState
from server.data_generator import (
    AccountSequence,
    generate_easy_episode,
    generate_medium_episode,
    generate_hard_episode,
)
from server.graders import (
    grade_easy,
    grade_medium,
    grade_hard_step,
    grade_hard_portfolio_bonus,
)


class FraudDetectionEnvironment:
    """
    Stateful fraud detection RL environment.

    Easy   — 5 independent single-txn sequences
    Medium — 1 eight-step single-account narrative
    Hard   — 3 five-step sequences interleaved (15 total)
    """

    def __init__(self) -> None:
        self._reset_state()

    # ──────────────────────────────────────────────────────────────────────────
    #  Public interface
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, task_level: str = "easy") -> FraudObservation:
        if task_level not in ("easy", "medium", "hard"):
            raise ValueError(f"task_level must be easy/medium/hard, got '{task_level}'")

        self._reset_state()
        self._episode_id = str(uuid.uuid4())
        seed = int(self._episode_id.replace("-", "")[:8], 16) % (2 ** 20)
        self._task_level = task_level

        if task_level == "easy":
            self._sequences = generate_easy_episode(seed)
            self._total_steps = sum(s.length for s in self._sequences)
        elif task_level == "medium":
            self._sequences = generate_medium_episode(seed)
            self._total_steps = self._sequences[0].length
        else:
            self._sequences = generate_hard_episode(seed)
            self._total_steps = sum(s.length for s in self._sequences)

        self._messages = ["Episode started. Analyse the transaction and classify it."]

        # Pull first transaction
        self._advance_sequence(prev_action=None)
        return self._build_observation(reward=None)

    def step(self, action: FraudAction) -> FraudObservation:
        if self._done:
            raise RuntimeError("Episode finished. Call reset() first.")

        txn = self._current_txn
        reward = self._compute_reward(action, txn)

        # Update per-step grader score
        if self._task_level == "easy":
            step_score = grade_easy(action, txn)
        elif self._task_level == "medium":
            step_score = grade_medium(
                action, txn, self._global_step, self._total_steps
            )
        else:
            step_score = grade_hard_step(action, txn)

        self._step_scores.append(step_score)

        # Track episode-level stats
        gt = txn["_ground_truth"]
        if action.classification == gt:
            self._correct_classifications += 1
        if gt == "legitimate" and action.classification == "fraud":
            self._false_positives += 1
        if gt == "fraud" and action.classification == "legitimate":
            self._false_negatives += 1

        # Store for portfolio bonus
        self._all_actions.append(action)
        self._all_txns.append(txn)

        self._messages = self._generate_feedback(action, txn, reward)
        self._global_step += 1

        # Advance to next transaction (reactive — passes agent's action)
        self._advance_sequence(prev_action=action.recommended_action)

        # Portfolio bonus at episode end for hard task
        if self._done and self._task_level == "hard":
            bonus = grade_hard_portfolio_bonus(self._all_actions, self._all_txns)
            reward = round(max(-1.0, min(1.0, reward + bonus)), 4)
            if bonus > 0:
                self._messages.append(
                    f"Portfolio bonus applied: +{bonus:.2f} "
                    f"(F1/recall/FPR thresholds met)"
                )

        return self._build_observation(reward=reward)

    @property
    def state(self) -> FraudState:
        score = (
            sum(self._step_scores) / len(self._step_scores)
            if self._step_scores else 0.0
        )
        complexity_map = {
            "easy":   "single_signal",
            "medium": "multi_signal_sequential",
            "hard":   "portfolio_level",
        }
        return FraudState(
            episode_id=self._episode_id,
            step_count=self._global_step,
            task_level=self._task_level,
            total_transactions=self._total_steps,
            correct_classifications=self._correct_classifications,
            false_positives=self._false_positives,
            false_negatives=self._false_negatives,
            current_score=round(score, 4),
            pattern_complexity=complexity_map.get(self._task_level, "unknown"),
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  Internal — state management
    # ──────────────────────────────────────────────────────────────────────────

    def _reset_state(self) -> None:
        """Full state wipe — zero leak between episodes."""
        self._episode_id: str = ""
        self._task_level: str = "easy"
        self._sequences: List[AccountSequence] = []
        self._seq_idx: int = 0          # which sequence we're currently drawing from
        self._global_step: int = 0
        self._total_steps: int = 0
        self._current_txn: Dict[str, Any] = {}
        self._done: bool = True
        self._correct_classifications: int = 0
        self._false_positives: int = 0
        self._false_negatives: int = 0
        self._step_scores: List[float] = []
        self._all_actions: List[FraudAction] = []
        self._all_txns: List[Dict[str, Any]] = []
        self._messages: List[str] = []
        self._step_in_sequence: Optional[int] = None
        self._sequence_length: Optional[int] = None

    def _advance_sequence(self, prev_action: Optional[str]) -> None:
        """
        Pull the next transaction from the current sequence (or next sequence).
        For hard: cycles through sequences round-robin.
        Sets self._done = True when all sequences are exhausted.
        """
        if self._task_level == "hard":
            # Round-robin across 3 sequences
            txn = self._next_from_round_robin(prev_action)
        else:
            txn = self._next_from_current(prev_action)

        if txn is None:
            self._done = True
        else:
            self._done = False
            self._current_txn = txn
            # Expose temporal context — which step within this account's sequence
            seq = self._sequences[min(self._seq_idx, len(self._sequences) - 1)]
            self._step_in_sequence = seq._step   # already incremented = 1-based
            self._sequence_length = seq.length

    def _next_from_current(self, prev_action: Optional[str]) -> Optional[Dict[str, Any]]:
        """For easy/medium: draw from sequences sequentially."""
        while self._seq_idx < len(self._sequences):
            seq = self._sequences[self._seq_idx]
            txn = seq.next(prev_action)
            if txn is not None:
                return txn
            self._seq_idx += 1
            prev_action = None  # reset between sequences
        return None

    def _next_from_round_robin(self, prev_action: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        For hard: interleave transactions across sequences round-robin.
        Each call picks from the sequence whose turn it is.
        """
        if not self._sequences:
            return None
        # Try each sequence starting from current idx
        attempts = 0
        while attempts < len(self._sequences):
            idx = self._seq_idx % len(self._sequences)
            seq = self._sequences[idx]
            txn = seq.next(prev_action if idx == self._seq_idx % len(self._sequences) else None)
            self._seq_idx += 1
            if txn is not None:
                return txn
            attempts += 1

        # All sequences exhausted
        return None

    # ──────────────────────────────────────────────────────────────────────────
    #  Reward function
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_reward(self, action: FraudAction, txn: Dict[str, Any]) -> float:
        """
        Shaped per-step reward. Clipped to [-1.0, 1.0].

        Penalty design:
          FP and FN penalties are MUTUALLY EXCLUSIVE and REPLACE the base
          wrong-classification penalty (-0.20) — they do not stack on top of it.
        """
        reward = 0.0
        gt = txn["_ground_truth"]
        gt_action = txn["_ground_truth_action"]
        gt_signals: List[str] = txn.get("_ground_truth_signals", [])
        correct = action.classification == gt

        # ── Classification (mutually exclusive) ───────────────────────────────
        if correct:
            reward += 0.35
        elif gt == "legitimate" and action.classification == "fraud":
            reward -= 0.30   # FP: replaces base -0.20
        elif gt == "fraud" and action.classification == "legitimate":
            reward -= 0.40   # FN: replaces base -0.20 (heaviest)
        else:
            reward -= 0.20   # other misclassification (e.g. suspicious↔fraud)

        # ── Flag-everything exploit prevention ────────────────────────────────
        if gt == "legitimate" and action.recommended_action == "flag_for_review":
            reward -= 0.15

        # ── Signal quality (proportional, only relevant signals) ──────────────
        if gt_signals:
            caught = len(set(action.triggered_signals) & set(gt_signals))
            reward += 0.20 * caught / len(gt_signals)
        else:
            # Legitimate: +0.10 for correctly raising no signals
            if not action.triggered_signals:
                reward += 0.10

        # ── Recommended action ────────────────────────────────────────────────
        if action.recommended_action == gt_action:
            reward += 0.25

        # ── Confidence calibration ────────────────────────────────────────────
        if correct and action.confidence > 0.70:
            reward += 0.10
        elif not correct and action.confidence > 0.80:
            reward -= 0.10   # overconfident + wrong

        return round(max(-1.0, min(1.0, reward)), 4)

    # ──────────────────────────────────────────────────────────────────────────
    #  Observation builder
    # ──────────────────────────────────────────────────────────────────────────

    def _build_observation(self, reward: Optional[float]) -> FraudObservation:
        txn = self._current_txn
        return FraudObservation(
            transaction_id=txn.get("transaction_id", "PENDING"),
            amount=txn.get("amount", 0.0),
            currency=txn.get("currency", "INR"),
            merchant_category=txn.get("merchant_category", ""),
            merchant_location=txn.get("merchant_location", ""),
            card_holder_location=txn.get("card_holder_location", ""),
            time_of_day=txn.get("time_of_day", "morning"),
            day_of_week=txn.get("day_of_week", "Monday"),
            transaction_history=txn.get("transaction_history", []),
            account_age_days=txn.get("account_age_days", 0),
            previous_fraud_flags=txn.get("previous_fraud_flags", 0),
            velocity_last_hour=txn.get("velocity_last_hour", 1),
            amount_vs_avg_ratio=txn.get("amount_vs_avg_ratio", 1.0),
            messages=list(self._messages),
            done=self._done,
            reward=reward,
            step_in_sequence=self._step_in_sequence,
            sequence_length=self._sequence_length,
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  Feedback generation
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_feedback(
        self, action: FraudAction, txn: Dict[str, Any], reward: float
    ) -> List[str]:
        msgs: List[str] = []
        gt = txn["_ground_truth"]
        gt_action = txn["_ground_truth_action"]
        gt_signals: List[str] = txn.get("_ground_truth_signals", [])

        if action.classification == gt:
            msgs.append(f"CORRECT: '{gt}'. Reward: {reward:+.2f}")
        else:
            msgs.append(
                f"INCORRECT: you said '{action.classification}', "
                f"actual was '{gt}'. Reward: {reward:+.2f}"
            )

        if action.recommended_action != gt_action:
            msgs.append(
                f"Wrong action: '{action.recommended_action}' → correct was '{gt_action}'"
            )

        if gt_signals:
            missed = sorted(set(gt_signals) - set(action.triggered_signals))
            extra  = sorted(set(action.triggered_signals) - set(gt_signals))
            if missed:
                msgs.append(f"Missed signals: {missed}")
            if extra:
                msgs.append(f"Spurious signals: {extra}")
        elif action.triggered_signals:
            msgs.append(
                f"Legitimate transaction — no signals should fire. "
                f"You raised: {action.triggered_signals}"
            )

        remaining = self._total_steps - self._global_step - 1
        if remaining > 0:
            msgs.append(f"{remaining} transaction(s) remaining.")
        else:
            avg = sum(self._step_scores) / len(self._step_scores) if self._step_scores else 0.0
            msgs.append(f"Episode complete. Average step score: {avg:.4f}")

        return msgs
