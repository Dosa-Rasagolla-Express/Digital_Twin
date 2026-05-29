"""
database.py
===========
SQLite database module for the Smart Traffic Digital Twin.
Handles schema creation, data insertion, and query helpers.
"""

import sqlite3
import os
from datetime import datetime

# ─────────────────────────────────────────────
# DATABASE PATH
# ─────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "traffic_twin.db")


def get_connection():
    """Return a thread-safe SQLite connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # Main traffic table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traffic (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            vehicle_count   INTEGER DEFAULT 0,
            avg_speed       REAL    DEFAULT 0.0,
            signal          TEXT    DEFAULT 'GREEN',
            congestion      TEXT    DEFAULT 'LOW',
            ambulance       INTEGER DEFAULT 0
        )
    """)

    # Junction-level traffic table for the Digital Twin map
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS junction_traffic (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            junction_name   TEXT    NOT NULL,
            vehicle_count   INTEGER DEFAULT 0,
            congestion      TEXT    DEFAULT 'LOW',
            signal          TEXT    DEFAULT 'GREEN'
        )
    """)

    # Alert log table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            alert_type  TEXT    NOT NULL,
            message     TEXT    NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# SAVE / INSERT FUNCTIONS
# ─────────────────────────────────────────────

def save_traffic_record(vehicle_count, avg_speed, signal, congestion, ambulance):
    """Insert a single traffic record into the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO traffic (timestamp, vehicle_count, avg_speed, signal, congestion, ambulance)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            int(vehicle_count),
            float(avg_speed),
            str(signal),
            str(congestion),
            int(ambulance),
        ),
    )
    conn.commit()
    conn.close()


def save_twin(twin):
    """Save a TrafficTwin object to the database (backwards compatible)."""
    save_traffic_record(
        twin.vehicle_count,
        twin.avg_speed,
        twin.signal_state,
        twin.congestion,
        int(twin.ambulance),
    )


def save_junction_record(junction_name, vehicle_count, congestion, signal):
    """Insert a junction traffic record."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO junction_traffic (timestamp, junction_name, vehicle_count, congestion, signal)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            junction_name,
            int(vehicle_count),
            str(congestion),
            str(signal),
        ),
    )
    conn.commit()
    conn.close()


def save_alert(alert_type, message):
    """Log an alert to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO alerts (timestamp, alert_type, message) VALUES (?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), alert_type, message),
    )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# QUERY HELPERS
# ─────────────────────────────────────────────

def fetch_all_traffic():
    """Return all traffic records as a list of dicts."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM traffic ORDER BY id ASC")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def fetch_latest_traffic():
    """Return the most recent traffic record."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM traffic ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def fetch_recent_alerts(limit=10):
    """Return the most recent alert records."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
    )
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def fetch_analytics():
    """Return aggregated analytics from the traffic table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            MAX(vehicle_count)      AS peak_traffic,
            AVG(vehicle_count)      AS avg_traffic,
            AVG(avg_speed)          AS avg_speed,
            SUM(ambulance)          AS emergency_count,
            COUNT(*)                AS total_records
        FROM traffic
    """)
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else {}


# ─────────────────────────────────────────────
# SEED DEMO DATA (if table is empty)
# ─────────────────────────────────────────────

def seed_demo_data():
    """Insert synthetic historical data for demonstration purposes."""
    import random
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM traffic")
    count = cursor.fetchone()[0]
    conn.close()

    if count >= 50:
        return  # Already seeded

    print("📦 Seeding demo traffic data …")
    signals = ["GREEN", "YELLOW", "RED"]
    congestions = ["LOW", "MEDIUM", "HIGH"]

    base_count = 5
    for i in range(120):
        # Simulate a realistic traffic wave
        hour_offset = i // 10
        peak_factor = 1 + 0.5 * abs(6 - hour_offset) / 6
        vc = max(1, int(base_count * peak_factor + random.randint(-2, 8)))
        spd = max(5.0, 60 - vc * 1.5 + random.uniform(-5, 5))
        sig = signals[i % 3]
        cong = (
            "LOW" if vc < 10
            else "MEDIUM" if vc < 20
            else "HIGH"
        )
        amb = 1 if random.random() < 0.05 else 0

        from datetime import timedelta
        ts = (datetime.now() - timedelta(minutes=120 - i)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        save_traffic_record(vc, spd, sig, cong, amb)


# ─────────────────────────────────────────────
# MODULE INIT
# ─────────────────────────────────────────────
init_db()
seed_demo_data()