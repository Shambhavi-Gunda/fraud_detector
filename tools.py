"""
"Investigation tools" the fraud agent can call while reasoning about a
transaction. In production these would hit real services (a geo-IP API,
a payments risk API, your bookings DB for velocity checks, an email
reputation service, etc). Here they run against the in-memory synthetic
dataset plus some static reference tables, so the whole thing runs
standalone with no external dependencies.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

DISPOSABLE_DOMAINS = {"mailinator.com", "tempmail.com", "guerrillamail.com", "10minutemail.com"}

# A crude "typical route price" reference table (median USD), used to flag
# bookings priced far outside the norm for that origin-destination pair.
ROUTE_PRICE_MEDIANS_DEFAULT = 450.0


class FraudTools:
    """Wraps tool functions and binds them to a dataset for velocity lookups."""

    def __init__(self, dataset: list[dict] | None = None):
        # dataset lets us simulate "look up this user's / device's other
        # bookings" the way a real service would query a bookings table.
        self.dataset = dataset or []

    # ---- tool implementations -------------------------------------------------

    def check_geo_mismatch(self, ip_country: str, billing_country: str, card_bin_country: str) -> dict[str, Any]:
        mismatches = []
        if ip_country != billing_country:
            mismatches.append(f"IP country ({ip_country}) != billing country ({billing_country})")
        if ip_country != card_bin_country:
            mismatches.append(f"IP country ({ip_country}) != card BIN country ({card_bin_country})")
        if billing_country != card_bin_country:
            mismatches.append(f"billing country ({billing_country}) != card BIN country ({card_bin_country})")
        return {
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "severity": "high" if len(mismatches) >= 2 else ("medium" if mismatches else "none"),
        }

    def check_email_reputation(self, email: str) -> dict[str, Any]:
        domain = email.split("@")[-1].lower()
        is_disposable = domain in DISPOSABLE_DOMAINS
        return {
            "domain": domain,
            "is_disposable": is_disposable,
            "risk": "high" if is_disposable else "low",
        }

    def check_account_history(
        self, account_age_days: int, prior_bookings_count: int, prior_chargebacks: int
    ) -> dict[str, Any]:
        is_new_account = account_age_days <= 3
        return {
            "account_age_days": account_age_days,
            "prior_bookings_count": prior_bookings_count,
            "prior_chargebacks": prior_chargebacks,
            "is_new_account": is_new_account,
            "has_chargeback_history": prior_chargebacks > 0,
            "risk": "high" if (is_new_account and prior_chargebacks > 0) else (
                "medium" if is_new_account or prior_chargebacks > 0 else "low"
            ),
        }

    def check_price_anomaly(self, origin: str, destination: str, price_usd: float) -> dict[str, Any]:
        # In production: look up a real median for this exact route.
        # Here we use a flat reference band and flag large deviations.
        median = ROUTE_PRICE_MEDIANS_DEFAULT
        deviation_pct = round((price_usd - median) / median * 100, 1)
        return {
            "route": f"{origin}-{destination}",
            "price_usd": price_usd,
            "reference_median_usd": median,
            "deviation_pct": deviation_pct,
            "risk": "high" if deviation_pct > 300 else ("medium" if deviation_pct > 100 else "low"),
        }

    def check_velocity(self, device_id: str, user_id: str, window_hours: int = 24) -> dict[str, Any]:
        """Look for other bookings sharing this device_id in a recent window,
        which suggests one actor operating multiple identities (account farming)."""
        if not self.dataset:
            return {"related_bookings_same_device": 0, "distinct_users_same_device": 0, "risk": "low"}

        same_device = [r for r in self.dataset if r["device_id"] == device_id]
        distinct_users = {r["user_id"] for r in same_device}
        related_count = len(same_device) - 1  # exclude the booking itself if present
        risk = "high" if len(distinct_users) >= 3 else ("medium" if len(distinct_users) == 2 else "low")
        return {
            "related_bookings_same_device": max(related_count, 0),
            "distinct_users_same_device": len(distinct_users),
            "risk": risk,
        }

    def check_booking_pattern(
        self, is_last_minute: bool, is_one_way: bool, num_travelers: int, booking_lead_time_days: int
    ) -> dict[str, Any]:
        red_flags = []
        if is_last_minute and is_one_way:
            red_flags.append("last-minute one-way booking (common in stolen-card cash-out fraud)")
        if booking_lead_time_days == 0:
            red_flags.append("same-day departure")
        return {
            "is_last_minute": is_last_minute,
            "is_one_way": is_one_way,
            "num_travelers": num_travelers,
            "red_flags": red_flags,
            "risk": "high" if red_flags else "low",
        }

    # ---- dispatch ---------------------------------------------------------

    def dispatch(self, tool_name: str, tool_input: dict) -> dict[str, Any]:
        fn = getattr(self, tool_name, None)
        if fn is None:
            return {"error": f"unknown tool '{tool_name}'"}
        return fn(**tool_input)


# ---- Anthropic tool-use schema definitions --------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "check_geo_mismatch",
        "description": "Check whether the IP country, billing country, and card BIN country agree. Mismatches are a strong signal of stolen payment credentials or VPN-masked fraud.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ip_country": {"type": "string"},
                "billing_country": {"type": "string"},
                "card_bin_country": {"type": "string"},
            },
            "required": ["ip_country", "billing_country", "card_bin_country"],
        },
    },
    {
        "name": "check_email_reputation",
        "description": "Check whether the booking email uses a disposable/throwaway domain, common among fraudsters avoiding traceability.",
        "input_schema": {
            "type": "object",
            "properties": {"email": {"type": "string"}},
            "required": ["email"],
        },
    },
    {
        "name": "check_account_history",
        "description": "Assess account trustworthiness: age, prior booking count, and prior chargebacks.",
        "input_schema": {
            "type": "object",
            "properties": {
                "account_age_days": {"type": "integer"},
                "prior_bookings_count": {"type": "integer"},
                "prior_chargebacks": {"type": "integer"},
            },
            "required": ["account_age_days", "prior_bookings_count", "prior_chargebacks"],
        },
    },
    {
        "name": "check_price_anomaly",
        "description": "Compare the booking price against a reference median for the route to flag suspiciously high-value bookings (common in card-testing / cash-out fraud).",
        "input_schema": {
            "type": "object",
            "properties": {
                "origin": {"type": "string"},
                "destination": {"type": "string"},
                "price_usd": {"type": "number"},
            },
            "required": ["origin", "destination", "price_usd"],
        },
    },
    {
        "name": "check_velocity",
        "description": "Check how many other bookings share this device fingerprint and how many distinct user identities used it. High reuse across identities suggests account farming or a fraud ring.",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_id": {"type": "string"},
                "user_id": {"type": "string"},
                "window_hours": {"type": "integer", "default": 24},
            },
            "required": ["device_id", "user_id"],
        },
    },
    {
        "name": "check_booking_pattern",
        "description": "Evaluate booking shape (last-minute, one-way, traveler count, lead time) for patterns typical of fraudulent bookings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "is_last_minute": {"type": "boolean"},
                "is_one_way": {"type": "boolean"},
                "num_travelers": {"type": "integer"},
                "booking_lead_time_days": {"type": "integer"},
            },
            "required": ["is_last_minute", "is_one_way", "num_travelers", "booking_lead_time_days"],
        },
    },
]
