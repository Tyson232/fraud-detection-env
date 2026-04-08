"""
Realistic Indian financial transaction data generator — hardened v2.

Key changes vs v1:
  - No single field directly exposes the answer
  - Amount ratios: 2x–4x (not 8x–12x) for fraud
  - Locations: neighbouring cities / same state (not foreign countries) for medium
  - Velocity: 3–4 txns over 2hrs (not 8 in 45min)
  - Legitimate txns have natural noise to trap rule-based agents
  - Every fraud pattern has at least one field that looks innocent
  - Hard patterns require reading across ALL fields + history to detect
"""
from __future__ import annotations

import uuid
import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# ── Signal vocabulary ─────────────────────────────────────────────────────────
SIGNAL_VOCAB = [
    "location_mismatch",
    "velocity_spike",
    "amount_anomaly",
    "unusual_time",
    "account_takeover_pattern",
    "card_testing_pattern",
    "new_device",
    "international_transaction",
    "high_risk_merchant",
    "chargeback_history",
    "money_mule_pattern",
    "synthetic_identity",
    "sudden_large_transaction",
    "no_prior_international_history",
    "rapid_merchant_change",
]

# ── Indian geography — state-wise neighbouring city pairs ─────────────────────
# Key insight: fraud often uses a NEARBY city, not a foreign country
CITY_NEIGHBOURS = {
    "Mumbai":    ["Pune", "Nashik", "Thane"],
    "Delhi":     ["Noida", "Gurgaon", "Faridabad"],
    "Bengaluru": ["Mysuru", "Mangaluru", "Hubli"],
    "Hyderabad": ["Secunderabad", "Warangal", "Vijayawada"],
    "Chennai":   ["Vellore", "Coimbatore", "Madurai"],
    "Kolkata":   ["Howrah", "Durgapur", "Asansol"],
    "Pune":      ["Mumbai", "Nashik", "Kolhapur"],
    "Ahmedabad": ["Surat", "Vadodara", "Rajkot"],
    "Jaipur":    ["Jodhpur", "Ajmer", "Kota"],
    "Lucknow":   ["Kanpur", "Agra", "Varanasi"],
}
INDIAN_CITIES = list(CITY_NEIGHBOURS.keys())

# Only used for hard sleeper fraud (genuine international)
FOREIGN_CITIES = [
    "Dubai, UAE", "Singapore", "Kuala Lumpur, Malaysia",
]

# Merchant fixtures
GROCERY      = ["DMart", "BigBazaar", "Reliance Fresh", "More Supermarket", "Spencer's"]
PHARMACY     = ["Apollo Pharmacy", "MedPlus", "Fortis Pharmacy", "Wellness Forever"]
FUEL         = ["Indian Oil Petrol Pump", "BPCL Fuel Station", "HP Petrol Bunk"]
ECOM         = ["Amazon India", "Flipkart", "Myntra", "Meesho", "Nykaa"]
RESTAURANT   = ["Haldiram's", "McDonald's India", "Domino's Pizza", "Café Coffee Day"]
ELECTRONICS  = ["Croma", "Reliance Digital", "Vijay Sales", "Samsung SmartCafé"]
LUXURY       = ["Tanishq", "Malabar Gold", "FabIndia Premium", "W for Woman"]
HIGH_RISK    = ["CoinSwitch Kuber", "WazirX Exchange", "BetWay India"]
APPAREL      = ["Westside", "Max Fashion", "Zara India", "H&M India"]
TIMES        = ["morning", "afternoon", "evening", "night"]
DAYS         = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _txn_id() -> str:
    return "TXN" + uuid.uuid4().hex[:10].upper()


def _ts(hours_ago: float = 0) -> str:
    return (datetime.now() - timedelta(hours=hours_ago)).isoformat()


def _neighbour(city: str, rng: random.Random) -> str:
    """Return a plausible neighbouring city — same state, not foreign."""
    return rng.choice(CITY_NEIGHBOURS.get(city, INDIAN_CITIES))


def _noisy_amount(base: float, rng: random.Random, lo: float = 0.80, hi: float = 1.20) -> float:
    """Add realistic noise to an amount — legitimate txns aren't perfectly round."""
    return round(base * rng.uniform(lo, hi) + rng.randint(0, 99), 2)


# ══════════════════════════════════════════════════════════════════════════════
#  Base class
# ══════════════════════════════════════════════════════════════════════════════

class AccountSequence(ABC):
    """
    Generates a coherent sequence of transactions for one account.
    The environment calls next(prev_action) at each step.

    Reactivity:
      'decline' on fraud → fraudster tries lower amount / different merchant
      'allow'   on fraud → fraudster grows slightly bolder
      'decline' on legit → customer retries same merchant
    """

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self._step = 0
        self._history: List[Dict[str, Any]] = []
        self._prev_declined = False
        self._boldness = 1.0

    def next(self, prev_action: Optional[str] = None) -> Optional[Dict[str, Any]]:
        self._react(prev_action)
        txn = self._generate_step(self._step)
        if txn is None:
            return None
        txn["transaction_id"] = _txn_id()
        txn["currency"] = "INR"
        txn["transaction_history"] = list(self._history[-5:])
        self._push_history(txn)
        self._step += 1
        return txn

    @property
    @abstractmethod
    def length(self) -> int:
        pass

    @abstractmethod
    def _generate_step(self, step: int) -> Optional[Dict[str, Any]]:
        pass

    def _react(self, prev_action: Optional[str]) -> None:
        if prev_action == "decline":
            self._prev_declined = True
            self._boldness = max(0.5, self._boldness * 0.7)
        elif prev_action == "allow":
            self._boldness = min(1.8, self._boldness * 1.10)

    def _push_history(self, txn: Dict[str, Any]) -> None:
        self._history.append({
            "transaction_id": txn["transaction_id"],
            "amount": txn["amount"],
            "location": txn["merchant_location"],
            "merchant": txn["merchant_category"],
            "timestamp": _ts(0),
        })


# ══════════════════════════════════════════════════════════════════════════════
#  EASY sequences — signals present but require combining 2-3 fields
#  No single field gives it away alone
# ══════════════════════════════════════════════════════════════════════════════

class LocationMismatchFraud(AccountSequence):
    """
    HARDER: Neighbouring city (not foreign), 2.5-4x amount (not 10x),
    evening not 3am. Rule agent won't fire on any single field alone.
    Agent must combine: location shift + amount anomaly + history shows no travel.
    """
    length = 1

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self._city = self.rng.choice(INDIAN_CITIES)
        self._avg = self.rng.uniform(3000, 7000)
        # History: all transactions in home city
        for i in range(5):
            self._history.append({
                "transaction_id": _txn_id(),
                "amount": _noisy_amount(self._avg, self.rng),
                "location": self._city,
                "merchant": self.rng.choice(GROCERY + RESTAURANT),
                "timestamp": _ts(self.rng.randint(48, 240)),
            })

    def _generate_step(self, step: int) -> Optional[Dict[str, Any]]:
        if step >= 1:
            return None
        # Neighbouring city — plausible but inconsistent with history
        fraud_city = _neighbour(self._city, self.rng)
        # Amount: 2.5-4x — unusual but not insane
        amt = round(self.rng.uniform(2.5, 4.0) * self._avg, 2)
        return {
            "amount": amt,
            "merchant_category": self.rng.choice(ELECTRONICS),
            "merchant_location": fraud_city,
            "card_holder_location": self._city,
            "time_of_day": "evening",          # not 3am — less obvious
            "day_of_week": self.rng.choice(DAYS[:5]),
            "account_age_days": self.rng.randint(400, 1500),
            "previous_fraud_flags": 0,
            "velocity_last_hour": 2,            # slightly elevated, not 8
            "amount_vs_avg_ratio": round(amt / self._avg, 2),
            "_ground_truth": "fraud",
            "_ground_truth_signals": [
                "location_mismatch", "amount_anomaly", "velocity_spike",
            ],
            "_ground_truth_action": "decline",
            "_pattern": "location_mismatch_fraud",
        }


class VelocityFraud(AccountSequence):
    """
    HARDER: 3-4 transactions in 2 hours (not 8 in 45min).
    Amounts vary naturally. Merchants are plausible (not all ecom).
    Agent must look at velocity + rapid_merchant_change together.
    """
    length = 1

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self._city = self.rng.choice(INDIAN_CITIES)
        self._avg = self.rng.uniform(1500, 4000)
        all_merchants = ECOM + RESTAURANT + PHARMACY + APPAREL
        for i in range(5):
            self._history.append({
                "transaction_id": _txn_id(),
                "amount": _noisy_amount(self._avg, self.rng, 0.5, 1.5),
                "location": self._city,
                "merchant": self.rng.choice(all_merchants),
                "timestamp": _ts(self.rng.randint(6, 72)),
            })

    def _generate_step(self, step: int) -> Optional[Dict[str, Any]]:
        if step >= 1:
            return None
        amt = round(self.rng.uniform(800, 2500), 2)
        return {
            "amount": amt,
            "merchant_category": self.rng.choice(ECOM + APPAREL),
            "merchant_location": self.rng.choice(
                [self._city] + CITY_NEIGHBOURS.get(self._city, [self._city])
            ),
            "card_holder_location": self._city,
            "time_of_day": self.rng.choice(["afternoon", "evening"]),
            "day_of_week": self.rng.choice(["Saturday", "Sunday"]),
            "account_age_days": self.rng.randint(90, 500),
            "previous_fraud_flags": 0,
            "velocity_last_hour": self.rng.randint(3, 5),   # 3-5, not 8
            "amount_vs_avg_ratio": round(amt / self._avg, 2),
            "_ground_truth": "fraud",
            "_ground_truth_signals": [
                "velocity_spike", "card_testing_pattern", "rapid_merchant_change",
            ],
            "_ground_truth_action": "decline",
            "_pattern": "velocity_fraud",
        }


class ObviousLegitimate(AccountSequence):
    """
    HARDER: Legitimate but with natural noise — slightly different city
    (within same metro), slightly higher amount than average, evening time.
    Rule agents that fire on ANY deviation will false-positive here.
    """
    length = 1

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self._city = self.rng.choice(INDIAN_CITIES)
        self._avg = self.rng.uniform(1000, 4000)
        for i in range(5):
            self._history.append({
                "transaction_id": _txn_id(),
                "amount": _noisy_amount(self._avg, self.rng),
                "location": self._city,
                "merchant": self.rng.choice(GROCERY + PHARMACY),
                "timestamp": _ts(self.rng.randint(24, 200)),
            })

    def _generate_step(self, step: int) -> Optional[Dict[str, Any]]:
        if step >= 1:
            return None
        # Noise: amount is 1.3-1.6x (slightly higher — maybe a special purchase)
        # but NOT suspicious
        amt = _noisy_amount(self._avg, self.rng, 1.1, 1.6)
        # Occasionally in a neighbouring city (weekend trip)
        use_neighbour = self.rng.random() < 0.4
        location = (
            _neighbour(self._city, self.rng) if use_neighbour else self._city
        )
        return {
            "amount": amt,
            "merchant_category": self.rng.choice(GROCERY + RESTAURANT + PHARMACY),
            "merchant_location": location,
            "card_holder_location": self._city,
            "time_of_day": self.rng.choice(["afternoon", "evening"]),
            "day_of_week": self.rng.choice(DAYS),
            "account_age_days": self.rng.randint(500, 2500),
            "previous_fraud_flags": 0,
            "velocity_last_hour": self.rng.randint(1, 2),
            "amount_vs_avg_ratio": round(amt / self._avg, 2),
            "_ground_truth": "legitimate",
            "_ground_truth_signals": [],
            "_ground_truth_action": "allow",
            "_pattern": "normal_grocery",
        }


# ══════════════════════════════════════════════════════════════════════════════
#  MEDIUM sequences — 8 steps, single account
#  No field alone triggers any simple rule
#  Pattern only visible across 3+ steps combined
# ══════════════════════════════════════════════════════════════════════════════

class AccountTakeoverSequence(AccountSequence):
    """
    HARDER:
    - Steps 1-3: completely normal (rule agent: allow)
    - Step 4: SAME STATE city (not foreign), amount 1.8x (not 5x), evening
              Each field alone: borderline normal
              Together with history: suspicious
    - Steps 5-8: fraud escalation in neighbouring city
    Agent must remember that steps 1-3 were in home city to detect step 4.
    """
    length = 8

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self._home = self.rng.choice(INDIAN_CITIES)
        self._attack_city = _neighbour(self._home, self.rng)
        self._avg = self.rng.uniform(2000, 5000)
        self._age = self.rng.randint(300, 900)

    def _generate_step(self, step: int) -> Optional[Dict[str, Any]]:
        if step >= 8:
            return None

        # Steps 0-2: normal home-city purchases
        if step < 3:
            amt = _noisy_amount(self._avg, self.rng)
            return {
                "amount": amt,
                "merchant_category": self.rng.choice(["grocery", "restaurant", "pharmacy"]),
                "merchant_location": self._home,
                "card_holder_location": self._home,
                "time_of_day": self.rng.choice(["morning", "afternoon", "evening"]),
                "day_of_week": self.rng.choice(DAYS[:5]),
                "account_age_days": self._age,
                "previous_fraud_flags": 0,
                "velocity_last_hour": 1,
                "amount_vs_avg_ratio": round(amt / self._avg, 2),
                "_ground_truth": "legitimate",
                "_ground_truth_signals": [],
                "_ground_truth_action": "allow",
                "_pattern": "normal_pre_takeover",
            }

        # Step 3: suspicious — neighbouring city, 1.8x amount, new merchant type
        if step == 3:
            amt = round(self.rng.uniform(1.6, 2.2) * self._avg, 2)
            return {
                "amount": amt,
                "merchant_category": self.rng.choice(ELECTRONICS),
                "merchant_location": self._attack_city,  # same state, not foreign
                "card_holder_location": self._home,
                "time_of_day": "evening",
                "day_of_week": self.rng.choice(DAYS[:5]),
                "account_age_days": self._age,
                "previous_fraud_flags": 0,
                "velocity_last_hour": 2,
                "amount_vs_avg_ratio": round(amt / self._avg, 2),
                "_ground_truth": "suspicious",
                "_ground_truth_signals": [
                    "account_takeover_pattern", "new_device",
                    "location_mismatch", "amount_anomaly",
                ],
                "_ground_truth_action": "request_verification",
                "_pattern": "account_takeover",
            }

        # Steps 4-7: active fraud — escalates unless declined
        if self._prev_declined:
            amt = round(self.rng.uniform(0.8, 1.5) * self._avg * self._boldness, 2)
            merchant = self.rng.choice(ECOM)
        else:
            amt = round(self.rng.uniform(2.5, 5.0) * self._avg * self._boldness, 2)
            merchant = self.rng.choice(ELECTRONICS + LUXURY)

        return {
            "amount": amt,
            "merchant_category": merchant,
            "merchant_location": self._attack_city,
            "card_holder_location": self._home,
            "time_of_day": "night" if step >= 6 else "evening",
            "day_of_week": self.rng.choice(DAYS),
            "account_age_days": self._age,
            "previous_fraud_flags": 0,
            "velocity_last_hour": min(5, step),
            "amount_vs_avg_ratio": round(amt / self._avg, 2),
            "_ground_truth": "fraud",
            "_ground_truth_signals": [
                "account_takeover_pattern", "location_mismatch",
                "amount_anomaly", "velocity_spike",
            ],
            "_ground_truth_action": "decline",
            "_pattern": "account_takeover",
        }


class FriendlyFraudSequence(AccountSequence):
    """
    HARDER:
    - Only 1 prior chargeback (not 3) — so chargeback_history alone isn't decisive
    - Amounts escalate gradually — no sudden jump
    - Location always home city — no location signal
    - Agent must combine: chargeback_history + gradual_escalation + merchant_type
    """
    length = 8

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self._city = self.rng.choice(INDIAN_CITIES)
        self._avg = self.rng.uniform(2000, 5000)
        self._age = self.rng.randint(500, 1500)
        self._chargebacks = 1   # only 1 — not an obvious signal alone

    def _generate_step(self, step: int) -> Optional[Dict[str, Any]]:
        if step >= 8:
            return None

        # Steps 0-2: normal, but account has 1 chargeback (rule agent ignores 1)
        if step < 3:
            amt = _noisy_amount(self._avg, self.rng)
            return {
                "amount": amt,
                "merchant_category": self.rng.choice(["grocery", "pharmacy", "restaurant"]),
                "merchant_location": self._city,
                "card_holder_location": self._city,
                "time_of_day": self.rng.choice(["morning", "afternoon"]),
                "day_of_week": self.rng.choice(DAYS[:5]),
                "account_age_days": self._age,
                "previous_fraud_flags": self._chargebacks,
                "velocity_last_hour": 1,
                "amount_vs_avg_ratio": round(amt / self._avg, 2),
                "_ground_truth": "legitimate",
                "_ground_truth_signals": [],
                "_ground_truth_action": "allow",
                "_pattern": "friendly_fraud_buildup",
            }

        # Steps 3-4: amounts creeping up — each step individually borderline
        if step < 5:
            # Gradual escalation: 1.5x, then 2.0x — not sudden 5x
            multiplier = 1.4 + (step - 3) * 0.4
            amt = round(multiplier * self._avg, 2)
            return {
                "amount": amt,
                "merchant_category": self.rng.choice(APPAREL + ELECTRONICS),
                "merchant_location": self._city,
                "card_holder_location": self._city,
                "time_of_day": self.rng.choice(["afternoon", "evening"]),
                "day_of_week": self.rng.choice(["Saturday", "Sunday"]),
                "account_age_days": self._age,
                "previous_fraud_flags": self._chargebacks,
                "velocity_last_hour": 1,
                "amount_vs_avg_ratio": round(amt / self._avg, 2),
                "_ground_truth": "suspicious",
                "_ground_truth_signals": ["chargeback_history", "amount_anomaly"],
                "_ground_truth_action": "flag_for_review",
                "_pattern": "friendly_fraud",
            }

        # Steps 5-7: clear fraud intent — high-value easily-resellable items
        amt = round(
            self.rng.uniform(3.5, 6.0) * self._avg * self._boldness, 2
        )
        return {
            "amount": amt,
            "merchant_category": self.rng.choice(LUXURY + ELECTRONICS),
            "merchant_location": self._city,
            "card_holder_location": self._city,
            "time_of_day": "afternoon",
            "day_of_week": "Saturday",
            "account_age_days": self._age,
            "previous_fraud_flags": self._chargebacks,
            "velocity_last_hour": 2,
            "amount_vs_avg_ratio": round(amt / self._avg, 2),
            "_ground_truth": "fraud",
            "_ground_truth_signals": ["chargeback_history", "amount_anomaly"],
            "_ground_truth_action": "decline",
            "_pattern": "friendly_fraud",
        }


class SleepFraudSequence(AccountSequence):
    """
    HARDER:
    - Account wakes up with amounts that are high but not insane (3-6x not 30-80x)
    - First international txn is to a common destination (Dubai, Singapore)
      not obviously high-risk (Lagos, Ghana)
    - Agent must read the HISTORY timestamps to spot the dormancy
    - Then combine: dormancy + first_international + moderate_amount_jump
    """
    length = 8

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self._city = self.rng.choice(INDIAN_CITIES)
        self._avg = self.rng.uniform(1500, 4000)
        self._age = self.rng.randint(700, 2000)
        # History: old timestamps — dormancy is only visible in history
        for i in range(5):
            self._history.append({
                "transaction_id": _txn_id(),
                "amount": _noisy_amount(self._avg, self.rng),
                "location": self._city,
                "merchant": self.rng.choice(GROCERY + PHARMACY),
                "timestamp": _ts(self.rng.randint(150, 210) * 24),  # 5-7 months ago
            })

    def _generate_step(self, step: int) -> Optional[Dict[str, Any]]:
        if step >= 8:
            return None

        # Steps 0-1: account waking up — local, slightly elevated
        if step < 2:
            amt = _noisy_amount(self._avg, self.rng, 1.0, 1.8)
            return {
                "amount": amt,
                "merchant_category": "grocery",
                "merchant_location": self._city,
                "card_holder_location": self._city,
                "time_of_day": "morning",
                "day_of_week": self.rng.choice(DAYS[:5]),
                "account_age_days": self._age,
                "previous_fraud_flags": 0,
                "velocity_last_hour": 1,
                "amount_vs_avg_ratio": round(amt / self._avg, 2),
                "_ground_truth": "suspicious",
                "_ground_truth_signals": ["sudden_large_transaction"],
                "_ground_truth_action": "flag_for_review",
                "_pattern": "sleeper_fraud",
            }

        # Steps 2-3: first international — common destination (not obviously risky)
        if step < 4:
            amt = round(self.rng.uniform(2.5, 5.0) * self._avg, 2)
            return {
                "amount": amt,
                "merchant_category": self.rng.choice(["luxury_retail", "electronics"]),
                "merchant_location": self.rng.choice(FOREIGN_CITIES),  # Dubai/Singapore
                "card_holder_location": self._city,
                "time_of_day": self.rng.choice(["morning", "afternoon"]),
                "day_of_week": self.rng.choice(DAYS[:5]),
                "account_age_days": self._age,
                "previous_fraud_flags": 0,
                "velocity_last_hour": 1,
                "amount_vs_avg_ratio": round(amt / self._avg, 2),
                "_ground_truth": "suspicious",
                "_ground_truth_signals": [
                    "no_prior_international_history",
                    "international_transaction",
                    "amount_anomaly",
                ],
                "_ground_truth_action": "request_verification",
                "_pattern": "sleeper_fraud",
            }

        # Steps 4-7: clear fraud — escalating
        amt = round(
            self.rng.uniform(5.0, 10.0) * self._avg * self._boldness, 2
        )
        return {
            "amount": amt,
            "merchant_category": "wire_transfer",
            "merchant_location": self.rng.choice(FOREIGN_CITIES),
            "card_holder_location": self._city,
            "time_of_day": "night",
            "day_of_week": self.rng.choice(DAYS),
            "account_age_days": self._age,
            "previous_fraud_flags": 0,
            "velocity_last_hour": step - 1,
            "amount_vs_avg_ratio": round(amt / self._avg, 2),
            "_ground_truth": "fraud",
            "_ground_truth_signals": [
                "sudden_large_transaction", "no_prior_international_history",
                "international_transaction", "amount_anomaly",
            ],
            "_ground_truth_action": "decline",
            "_pattern": "sleeper_fraud",
        }


# ══════════════════════════════════════════════════════════════════════════════
#  HARD sequences — no single field triggers any rule
#  Only full cross-field + history reasoning reveals fraud
# ══════════════════════════════════════════════════════════════════════════════

class MoneyMuleSequence(AccountSequence):
    """
    HARDER:
    - Amounts look like normal salary/EMI transfers (₹8k-₹25k range)
    - Same bank transfer merchant (not obviously different)
    - Pattern only visible as: IN → OUT → IN → OUT cycle in history
    - Individual txns: perfectly normal fund transfers
    - Agent must read history to spot the cycling pattern
    """
    length = 5

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self._city = self.rng.choice(INDIAN_CITIES)
        self._age = self.rng.randint(180, 500)
        self._base = self.rng.randint(8000, 25000)

    def _generate_step(self, step: int) -> Optional[Dict[str, Any]]:
        if step >= 5:
            return None
        is_deposit = step % 2 == 0
        # Slight variation to look like real transfers
        amt = float(self._base + self.rng.randint(-500, 500))
        if not is_deposit:
            amt = amt - self.rng.randint(200, 800)  # slightly less than deposit

        merchant = "NEFT Transfer IN" if is_deposit else "IMPS Transfer OUT"
        dest = self._city if is_deposit else _neighbour(self._city, self.rng)
        # Individual txns look normal — pattern only visible in sequence
        gt = "suspicious" if step < 2 else "fraud"
        ratio = round(amt / self._base, 2)

        return {
            "amount": round(amt * self._boldness, 2),
            "merchant_category": "fund_transfer",
            "merchant_location": dest,
            "card_holder_location": self._city,
            "time_of_day": self.rng.choice(["morning", "afternoon"]),
            "day_of_week": self.rng.choice(DAYS[:5]),
            "account_age_days": self._age,
            "previous_fraud_flags": 0,
            "velocity_last_hour": step + 1,
            "amount_vs_avg_ratio": ratio,
            "_ground_truth": gt,
            "_ground_truth_signals": [
                "money_mule_pattern", "velocity_spike", "rapid_merchant_change",
            ],
            "_ground_truth_action": "decline" if gt == "fraud" else "flag_for_review",
            "_pattern": "money_mule",
        }


class SyntheticIdentitySequence(AccountSequence):
    """
    HARDER:
    - Account is 6 months old (not suspiciously new — 180-220 days)
    - Steps 0-2: legitimate small purchases that build a real-looking history
    - Step 3: first large purchase is 4x avg (not 60x) — plausible as a one-time buy
    - Step 4: second large purchase makes pattern clear
    - Agent must read account_age + escalation pattern in history to decide
    """
    length = 5

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self._city = self.rng.choice(INDIAN_CITIES)
        self._age = self.rng.randint(180, 220)
        self._base = self.rng.uniform(600, 1500)

    def _generate_step(self, step: int) -> Optional[Dict[str, Any]]:
        if step >= 5:
            return None
        # Escalating base
        self._base *= self.rng.uniform(1.05, 1.20)

        if step < 3:
            amt = _noisy_amount(self._base, self.rng)
            return {
                "amount": amt,
                "merchant_category": self.rng.choice(["grocery", "pharmacy", "restaurant"]),
                "merchant_location": self._city,
                "card_holder_location": self._city,
                "time_of_day": self.rng.choice(["morning", "afternoon"]),
                "day_of_week": self.rng.choice(DAYS[:5]),
                "account_age_days": self._age,
                "previous_fraud_flags": 0,
                "velocity_last_hour": 1,
                "amount_vs_avg_ratio": round(self.rng.uniform(0.9, 1.2), 2),
                "_ground_truth": "legitimate",
                "_ground_truth_signals": [],
                "_ground_truth_action": "allow",
                "_pattern": "synthetic_buildup",
            }

        # Steps 3-4: sudden jump — 4-8x the escalated base
        amt = round(
            self.rng.uniform(4.0, 8.0) * self._base * self._boldness, 2
        )
        gt = "suspicious" if step == 3 else "fraud"
        return {
            "amount": amt,
            "merchant_category": self.rng.choice(LUXURY + ELECTRONICS),
            "merchant_location": self._city,
            "card_holder_location": self._city,
            "time_of_day": self.rng.choice(["afternoon", "evening"]),
            "day_of_week": self.rng.choice(["Friday", "Saturday"]),
            "account_age_days": self._age,
            "previous_fraud_flags": 0,
            "velocity_last_hour": 1,
            "amount_vs_avg_ratio": round(amt / self._base, 2),
            "_ground_truth": gt,
            "_ground_truth_signals": [
                "synthetic_identity", "sudden_large_transaction", "amount_anomaly",
            ],
            "_ground_truth_action": "request_verification" if gt == "suspicious" else "decline",
            "_pattern": "synthetic_identity",
        }


class LegitAccountSequence(AccountSequence):
    """
    HARDER trap:
    - Has natural noise: amounts vary 0.7x-1.8x, occasional neighbouring city
    - Sometimes evening purchases, occasional slight velocity (2 txns/hr)
    - Rule agents that fire on ANY of these will false-positive
    - Only an agent that weighs ALL signals together will correctly allow
    """
    length = 5

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self._city = self.rng.choice(INDIAN_CITIES)
        self._avg = self.rng.uniform(1000, 5000)
        self._age = self.rng.randint(700, 3000)

    def _generate_step(self, step: int) -> Optional[Dict[str, Any]]:
        if step >= 5:
            return None
        cats = ["grocery", "restaurant", "pharmacy", "fuel", "ecommerce"]
        cat = cats[step % len(cats)]
        amt = _noisy_amount(self._avg, self.rng, 0.65, 1.75)

        # Natural noise: 40% chance of neighbouring city (weekend trip / work travel)
        use_neighbour = self.rng.random() < 0.4
        loc = _neighbour(self._city, self.rng) if use_neighbour else self._city

        # Natural noise: occasional slight velocity (2 txns this hour)
        velocity = self.rng.choice([1, 1, 1, 2])

        return {
            "amount": amt,
            "merchant_category": cat,
            "merchant_location": loc,
            "card_holder_location": self._city,
            "time_of_day": self.rng.choice(["morning", "afternoon", "evening"]),
            "day_of_week": self.rng.choice(DAYS),
            "account_age_days": self._age,
            "previous_fraud_flags": 0,
            "velocity_last_hour": velocity,
            "amount_vs_avg_ratio": round(amt / self._avg, 2),
            "_ground_truth": "legitimate",
            "_ground_truth_signals": [],
            "_ground_truth_action": "allow",
            "_pattern": "normal",
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Episode builders
# ══════════════════════════════════════════════════════════════════════════════

def generate_easy_episode(seed: int = 0) -> List[AccountSequence]:
    """5 single-step sequences: requires combining 2-3 fields, no single-field answer."""
    rng = random.Random(seed)
    pool = [
        LocationMismatchFraud(seed + 1),
        VelocityFraud(seed + 2),
        ObviousLegitimate(seed + 3),
        LocationMismatchFraud(seed + 4),
        ObviousLegitimate(seed + 5),
    ]
    rng.shuffle(pool)
    return pool


def generate_medium_episode(seed: int = 0) -> List[AccountSequence]:
    """One 8-step single-account narrative. Pattern invisible until step 3-4."""
    rng = random.Random(seed)
    choice = rng.randint(0, 2)
    if choice == 0:
        return [AccountTakeoverSequence(seed)]
    elif choice == 1:
        return [FriendlyFraudSequence(seed)]
    else:
        return [SleepFraudSequence(seed)]


def generate_hard_episode(seed: int = 0) -> List[AccountSequence]:
    """
    3 interleaved account sequences — 15 total transactions.
      MoneyMuleSequence:         5 txns (individually normal, pattern = fraud)
      SyntheticIdentitySequence: 5 txns (2 legit buildup + 1 suspicious + 2 fraud)
      LegitAccountSequence:      5 txns (noisy legitimate — traps rule agents)
    """
    return [
        MoneyMuleSequence(seed + 10),
        SyntheticIdentitySequence(seed + 20),
        LegitAccountSequence(seed + 30),
    ]
