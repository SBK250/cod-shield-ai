"""SQLite database for COD Shield AI incident reports."""

from collections.abc import Iterator
from contextlib import contextmanager
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "data"
DATABASE_FILE = DATABASE_DIR / "cod_shield.db"


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Create a database connection."""

    DATABASE_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row

    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database():
    """Create the incident table if it does not exist."""

    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS incident_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_reference TEXT NOT NULL,
                platform TEXT NOT NULL,
                incident_type TEXT NOT NULL,
                description TEXT NOT NULL,
                reported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL DEFAULT 'Submitted'
            )
            """
        )


def create_incident(
    order_reference,
    platform,
    incident_type,
    description,
):
    """Save an incident report."""

    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO incident_reports (
                order_reference,
                platform,
                incident_type,
                description
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                order_reference,
                platform,
                incident_type,
                description,
            ),
        )

        return cursor.lastrowid


def get_incidents():
    """Retrieve incident reports, newest first."""

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                order_reference,
                platform,
                incident_type,
                description,
                reported_at,
                status
            FROM incident_reports
            ORDER BY id DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]