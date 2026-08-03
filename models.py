"""
Pydantic schemas shared across the app: the shape of a booking transaction
going IN to the agent, and the shape of the fraud verdict coming OUT.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class BookingTransaction(BaseModel):
    booking_id: str
    user_id: str
    created_at: datetime
    email: str
    origin: str
    destination: str
    departure_date: datetime
    booking_lead_time_days: int = Field(..., description="Days between booking and departure")
    price_usd: float
    currency: str = "USD"
    payment_method: str  # credit_card, debit_card, paypal, wallet
    card_bin_country: str  # country implied by card BIN
    billing_country: str
    ip_country: str
    device_id: str
    device_fingerprint_reused: bool = False
    num_travelers: int
    account_age_days: int
    prior_bookings_count: int
    prior_chargebacks: int
    is_one_way: bool
    is_last_minute: bool


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolCallLog(BaseModel):
    tool: str
    input: dict
    output: dict


class FraudVerdict(BaseModel):
    booking_id: str
    risk_score: int = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    flags: list[str]
    explanation: str
    recommended_action: str
    tool_calls: list[ToolCallLog] = []
