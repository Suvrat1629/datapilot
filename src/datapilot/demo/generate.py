"""Synthetic Ola-style ride-booking data. Seeded, so results are reproducible.

Known patterns injected for the eval set:
  - Bengaluru has a much higher cancellation rate on weekends.
  - Mumbai has the highest average fare.
  - "driver_unavailable" is the top cancel reason overall.
  - Cash is the most used payment method.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import psycopg

SEED = 1629
N_CUSTOMERS = 5000
N_DRIVERS = 600
N_BOOKINGS = 50_000
DAYS = 90

CITIES = [
    ("Bengaluru", "Karnataka"),
    ("Mumbai", "Maharashtra"),
    ("Delhi", "Delhi"),
    ("Hyderabad", "Telangana"),
    ("Chennai", "Tamil Nadu"),
    ("Kolkata", "West Bengal"),
    ("Pune", "Maharashtra"),
    ("Ahmedabad", "Gujarat"),
    ("Jaipur", "Rajasthan"),
    ("Lucknow", "Uttar Pradesh"),
    ("Chandigarh", "Chandigarh"),
    ("Kochi", "Kerala"),
]
CITY_WEIGHT = [14, 14, 14, 10, 9, 8, 8, 6, 5, 4, 4, 4]
FARE_PER_KM = {"Mumbai": 22.0, "Delhi": 18.0, "Bengaluru": 17.0}
DEFAULT_FARE_PER_KM = 14.0
VEHICLES = ["auto", "mini", "sedan", "bike"]
VEHICLE_MULT = {"auto": 0.8, "mini": 1.0, "sedan": 1.4, "bike": 0.5}
CANCEL_REASONS = ["driver_unavailable", "customer_changed_mind", "wrong_address", "long_wait"]
CANCEL_REASON_W = [45, 25, 12, 18]
PAY_METHODS = ["cash", "upi", "card", "wallet"]
PAY_W = [40, 35, 15, 10]

FIRST = [
    "Aarav",
    "Vivaan",
    "Aditya",
    "Diya",
    "Ananya",
    "Ishaan",
    "Kavya",
    "Rohan",
    "Meera",
    "Arjun",
    "Saanvi",
    "Karan",
    "Neha",
    "Rahul",
    "Priya",
    "Vikram",
    "Sneha",
    "Amit",
]
LAST = ["Sharma", "Verma", "Iyer", "Reddy", "Patel", "Nair", "Gupta", "Singh", "Das", "Mehta"]

DEMO_TABLES = frozenset({"cities", "customers", "drivers", "bookings", "payments"})
MARKER = "_datapilot_demo"


class NotDemoDatabase(Exception):
    """`demo load` drops and recreates tables, so it refuses to run on someone's real data."""


def check_demo_target(admin: psycopg.Connection, force: bool = False) -> None:
    rows = admin.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
    ).fetchall()
    names = {r["table_name"] for r in rows}
    if force or not names or MARKER in names:
        return
    clash = sorted(names & DEMO_TABLES)
    found = ", ".join(sorted(names)[:8])
    raise NotDemoDatabase(
        f"This database already has tables in the public schema ({found}) and was not created "
        f"by `demo load`. Loading the demo would DROP: {', '.join(sorted(DEMO_TABLES))}"
        + (f" (existing: {', '.join(clash)})" if clash else "")
        + ". Use an empty database, or pass --force if you are sure these are demo tables."
    )


DDL = f"""
DROP TABLE IF EXISTS payments, bookings, drivers, customers, cities, {MARKER} CASCADE;
CREATE TABLE {MARKER} (created_at timestamptz NOT NULL DEFAULT now());
INSERT INTO {MARKER} DEFAULT VALUES;

CREATE TABLE cities (
    id    integer PRIMARY KEY,
    name  text NOT NULL UNIQUE,
    state text NOT NULL
);
CREATE TABLE customers (
    id      integer PRIMARY KEY,
    name    text NOT NULL,
    phone   text NOT NULL,
    email   text NOT NULL,
    city_id integer NOT NULL REFERENCES cities(id)
);
CREATE TABLE drivers (
    id           integer PRIMARY KEY,
    name         text NOT NULL,
    rating       numeric(2,1) NOT NULL,
    city_id      integer NOT NULL REFERENCES cities(id),
    vehicle_type text NOT NULL
);
CREATE TABLE bookings (
    id            integer PRIMARY KEY,
    customer_id   integer NOT NULL REFERENCES customers(id),
    driver_id     integer REFERENCES drivers(id),
    city_id       integer NOT NULL REFERENCES cities(id),
    vehicle_type  text NOT NULL,
    status        text NOT NULL,
    cancel_reason text,
    distance_km   numeric(6,2) NOT NULL,
    fare_amount   numeric(10,2),
    created_at    timestamptz NOT NULL
);
CREATE TABLE payments (
    id         integer PRIMARY KEY,
    booking_id integer NOT NULL REFERENCES bookings(id),
    method     text NOT NULL,
    amount     numeric(10,2) NOT NULL,
    paid_at    timestamptz NOT NULL
);
CREATE INDEX ON bookings (created_at);
CREATE INDEX ON bookings (city_id, status);
"""


def _cancel_probability(city: str, created: datetime) -> float:
    base = 0.14
    if city == "Bengaluru" and created.weekday() >= 5:
        return 0.38
    return base


def generate(admin: psycopg.Connection) -> dict[str, int]:
    rng = random.Random(SEED)
    admin.execute(DDL)

    with admin.cursor() as cur:
        with cur.copy("COPY cities (id, name, state) FROM STDIN") as cp:
            for i, (name, state) in enumerate(CITIES, start=1):
                cp.write_row((i, name, state))

        city_ids = list(range(1, len(CITIES) + 1))
        with cur.copy("COPY customers (id, name, phone, email, city_id) FROM STDIN") as cp:
            for i in range(1, N_CUSTOMERS + 1):
                name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
                cp.write_row(
                    (
                        i,
                        name,
                        f"+91{rng.randint(7_000_000_000, 9_999_999_999)}",
                        f"user{i}@example.com",
                        rng.choices(city_ids, CITY_WEIGHT)[0],
                    )
                )

        drivers_by_city: dict[int, list[int]] = {c: [] for c in city_ids}
        with cur.copy("COPY drivers (id, name, rating, city_id, vehicle_type) FROM STDIN") as cp:
            for i in range(1, N_DRIVERS + 1):
                city = rng.choices(city_ids, CITY_WEIGHT)[0]
                drivers_by_city[city].append(i)
                cp.write_row(
                    (
                        i,
                        f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                        round(rng.uniform(3.4, 5.0), 1),
                        city,
                        rng.choice(VEHICLES),
                    )
                )

        now = datetime.now(UTC).replace(microsecond=0)
        payments: list[tuple] = []
        pay_id = 0
        n_completed = n_cancelled = 0
        with cur.copy(
            "COPY bookings (id, customer_id, driver_id, city_id, vehicle_type, status, "
            "cancel_reason, distance_km, fare_amount, created_at) FROM STDIN"
        ) as cp:
            for i in range(1, N_BOOKINGS + 1):
                city_id = rng.choices(city_ids, CITY_WEIGHT)[0]
                city = CITIES[city_id - 1][0]
                created = now - timedelta(seconds=rng.randint(0, DAYS * 86400))
                vehicle = rng.choice(VEHICLES)
                distance = round(max(0.8, rng.lognormvariate(1.9, 0.6)), 2)
                roll = rng.random()
                cancel_p = _cancel_probability(city, created)
                if roll < cancel_p:
                    status, reason, fare = (
                        "cancelled",
                        rng.choices(CANCEL_REASONS, CANCEL_REASON_W)[0],
                        None,
                    )
                    driver = (
                        rng.choice(drivers_by_city[city_id])
                        if (reason != "driver_unavailable" and drivers_by_city[city_id])
                        else None
                    )
                    n_cancelled += 1
                elif roll < cancel_p + 0.03:
                    status, reason, fare, driver = "no_show", None, None, None
                else:
                    status, reason = "completed", None
                    per_km = FARE_PER_KM.get(city, DEFAULT_FARE_PER_KM)
                    fare = round(
                        35 + distance * per_km * VEHICLE_MULT[vehicle] * rng.uniform(0.9, 1.15), 2
                    )
                    driver = (
                        rng.choice(drivers_by_city[city_id]) if drivers_by_city[city_id] else None
                    )
                    n_completed += 1
                    pay_id += 1
                    payments.append(
                        (
                            pay_id,
                            i,
                            rng.choices(PAY_METHODS, PAY_W)[0],
                            fare,
                            created + timedelta(minutes=rng.randint(10, 60)),
                        )
                    )
                cp.write_row(
                    (
                        i,
                        rng.randint(1, N_CUSTOMERS),
                        driver,
                        city_id,
                        vehicle,
                        status,
                        reason,
                        distance,
                        fare,
                        created,
                    )
                )

        with cur.copy("COPY payments (id, booking_id, method, amount, paid_at) FROM STDIN") as cp:
            for row in payments:
                cp.write_row(row)

    admin.commit()
    return {
        "cities": len(CITIES),
        "customers": N_CUSTOMERS,
        "drivers": N_DRIVERS,
        "bookings": N_BOOKINGS,
        "completed": n_completed,
        "cancelled": n_cancelled,
        "payments": len(payments),
    }
