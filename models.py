from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, field_validator


class FraudAction(BaseModel):
    classification: str
    # must be one of: "legitimate", "suspicious", "fraud"

    confidence: float
    # 0.0 to 1.0

    triggered_signals: List[str]
    # list of signal names the agent flagged
    # e.g. ["location_mismatch", "velocity_spike", "amount_anomaly"]

    recommended_action: str
    # one of: "allow", "flag_for_review", "decline", "request_verification"

    reasoning: str
    # one sentence explanation

    @field_validator("classification")
    @classmethod
    def validate_classification(cls, v: str) -> str:
        allowed = {"legitimate", "suspicious", "fraud"}
        if v not in allowed:
            raise ValueError(f"classification must be one of {allowed}, got '{v}'")
        return v

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {v}")
        return round(v, 4)

    @field_validator("recommended_action")
    @classmethod
    def validate_recommended_action(cls, v: str) -> str:
        allowed = {"allow", "flag_for_review", "decline", "request_verification"}
        if v not in allowed:
            raise ValueError(f"recommended_action must be one of {allowed}, got '{v}'")
        return v


class FraudObservation(BaseModel):
    transaction_id: str
    amount: float
    currency: str
    merchant_category: str
    merchant_location: str
    card_holder_location: str
    time_of_day: str          # "morning", "afternoon", "evening", "night"
    day_of_week: str
    transaction_history: List[dict]
    # last 5 transactions with amount, location, merchant, timestamp
    account_age_days: int
    previous_fraud_flags: int
    velocity_last_hour: int   # number of transactions in last hour
    amount_vs_avg_ratio: float  # current amount / 30 day average
    messages: List[str]       # feedback messages from environment
    done: bool
    reward: Optional[float] = None
    step_in_sequence: Optional[int] = None   # position within current account sequence (1-based)
    sequence_length: Optional[int] = None    # total steps in current account sequence


class FraudState(BaseModel):
    episode_id: str
    step_count: int
    task_level: str           # "easy", "medium", "hard"
    total_transactions: int
    correct_classifications: int
    false_positives: int
    false_negatives: int
    current_score: float
    pattern_complexity: str
