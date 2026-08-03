"""
Generates a realistic synthetic dataset of travel booking transactions.

A fraction of transactions are seeded with classic fraud patterns so the
agent has real signal to detect:
  - IP / billing / card-BIN country mismatches (geo mismatch)
  - Brand-new accounts making expensive, last-minute, one-way bookings
  - Booking velocity: many bookings in a short window from the same
    device/IP but with different traveler identities
  - Disposable / throwaway email domains
  - Prior chargeback history
  - Reused device fingerprints across "different" users (account farming)
"""
from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timedelta

from faker import Faker

fake = Faker()

COUNTRIES = ["US", "GB", "DE", "FR", "NG", "IN", "BR", "RU", "CN", "AE", "PH", "VN", "ZA", "MX", "PK"]
HIGH_RISK_CORRIDOR_COUNTRIES = ["NG", "PK", "VN", "RU"]  # used only to bias synthetic fraud generation
AIRPORTS = ["JFK", "LHR", "CDG", "FRA", "DXB", "SIN", "LOS", "DEL", "GRU", "MEX", "JNB", "SVO", "PEK", "MNL", "HAN"]
PAYMENT_METHODS = ["credit_card", "debit_card", "paypal", "wallet"]
DISPOSABLE_DOMAINS = ["mailinator.com", "tempmail.com", "guerrillamail.com", "10minutemail.com"]
NORMAL_DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "icloud.com", "protonmail.com"]


def _rand_country_pair(mismatch: bool) -> tuple[str, str, str]:
    ip = random.choice(COUNTRIES)
    if mismatch:
        billing = random.choice([c for c in COUNTRIES if c != ip])
        card = random.choice([c for c in COUNTRIES if c != ip])
    else:
        billing = ip
        card = ip
    return ip, billing, card


def _make_transaction(is_fraud: bool, shared_device_id: str | None = None) -> dict:
    booking_id = str(uuid.uuid4())
    user_id = str(uuid.uuid4())
    created_at = fake.date_time_between(start_date="-30d", end_date="now")

    if is_fraud:
        # Fraud transactions skew: new accounts, geo mismatches, last-minute,
        # one-way, expensive, disposable email, occasionally reused device.
        account_age_days = random.randint(0, 3)
        prior_bookings_count = 0
        prior_chargebacks = random.choice([0, 0, 1, 2])
        lead_time = random.randint(0, 2)
        is_last_minute = True
        is_one_way = random.random() < 0.7
        price = round(random.uniform(900, 4500), 2)
        ip_country, billing_country, card_bin_country = _rand_country_pair(mismatch=random.random() < 0.75)
        email_domain = random.choice(DISPOSABLE_DOMAINS) if random.random() < 0.5 else random.choice(NORMAL_DOMAINS)
        device_id = shared_device_id or str(uuid.uuid4())
        device_reused = shared_device_id is not None
        payment_method = random.choice(["credit_card", "debit_card"])
    else:
        account_age_days = random.randint(30, 2500)
        prior_bookings_count = random.randint(1, 40)
        prior_chargebacks = 0
        lead_time = random.randint(3, 180)
        is_last_minute = lead_time <= 2
        is_one_way = random.random() < 0.15
        price = round(random.uniform(80, 1200), 2)
        ip_country, billing_country, card_bin_country = _rand_country_pair(mismatch=random.random() < 0.03)
        email_domain = random.choice(NORMAL_DOMAINS)
        device_id = str(uuid.uuid4())
        device_reused = False
        payment_method = random.choice(PAYMENT_METHODS)

    origin, destination = random.sample(AIRPORTS, 2)
    departure_date = created_at + timedelta(days=lead_time)

    return {
        "booking_id": booking_id,
        "user_id": user_id,
        "created_at": created_at.isoformat(),
        "email": f"{fake.user_name()}@{email_domain}",
        "origin": origin,
        "destination": destination,
        "departure_date": departure_date.isoformat(),
        "booking_lead_time_days": lead_time,
        "price_usd": price,
        "currency": "USD",
        "payment_method": payment_method,
        "card_bin_country": card_bin_country,
        "billing_country": billing_country,
        "ip_country": ip_country,
        "device_id": device_id,
        "device_fingerprint_reused": device_reused,
        "num_travelers": random.randint(1, 4),
        "account_age_days": account_age_days,
        "prior_bookings_count": prior_bookings_count,
        "prior_chargebacks": prior_chargebacks,
        "is_one_way": is_one_way,
        "is_last_minute": is_last_minute,
        "label": "fraud" if is_fraud else "legit",  # ground truth, NOT shown to the agent
    }


def generate_dataset(n: int = 200, fraud_rate: float = 0.12, seed: int | None = 42) -> list[dict]:
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)

    n_fraud = max(1, int(n * fraud_rate))
    n_legit = n - n_fraud

    records = [_make_transaction(is_fraud=False) for _ in range(n_legit)]

    # Sprinkle in a few "device farming" clusters: same device_id, several
    # different user identities, booked in quick succession -> velocity fraud.
    fraud_records = []
    remaining_fraud = n_fraud
    while remaining_fraud > 0:
        cluster_size = min(remaining_fraud, random.choice([1, 1, 1, 2, 3]))
        shared_device = str(uuid.uuid4()) if cluster_size > 1 else None
        for _ in range(cluster_size):
            fraud_records.append(_make_transaction(is_fraud=True, shared_device_id=shared_device))
        remaining_fraud -= cluster_size

    records.extend(fraud_records)
    random.shuffle(records)
    return records


if __name__ == "__main__":
    data = generate_dataset(n=200, fraud_rate=0.12)
    with open("synthetic_bookings.json", "w") as f:
        json.dump(data, f, indent=2)
    n_fraud = sum(1 for r in data if r["label"] == "fraud")
    print(f"Generated {len(data)} transactions ({n_fraud} fraud, {len(data) - n_fraud} legit) -> synthetic_bookings.json")
