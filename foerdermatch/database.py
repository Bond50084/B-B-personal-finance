"""Component B — SQLite persistence for client profiles and run logs.

The standard-library ``sqlite3`` module is sufficient for the MVP. All SQL
lives in this single module behind small CRUD functions, so the planned
upgrade to PostgreSQL (psycopg / SQLAlchemy) only touches this file — the
rest of the application works with plain dictionaries.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional

try:  # works both as a package (Flask) and standalone (python main.py)
    from .config import Settings
    from .utils import NATIONWIDE, canonical_region, utcnow_iso
except ImportError:
    from config import Settings
    from utils import NATIONWIDE, canonical_region, utcnow_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
    client_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name        TEXT NOT NULL,
    industry            TEXT NOT NULL DEFAULT '',
    state               TEXT NOT NULL,                 -- canonical Bundesland or 'bundesweit'
    company_size        TEXT NOT NULL DEFAULT 'KMU',   -- free text, e.g. 'KMU, 80 Mitarbeitende'
    project_description TEXT NOT NULL,                 -- Projektbeschreibung (RAG query text)
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

-- One row per ingestion run: powers the "daily update" audit trail.
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at      TEXT NOT NULL,
    source      TEXT NOT NULL,                          -- 'mock' | 'live'
    n_total     INTEGER NOT NULL,
    n_new       INTEGER NOT NULL,
    n_updated   INTEGER NOT NULL,
    n_unchanged INTEGER NOT NULL,
    n_errors    INTEGER NOT NULL DEFAULT 0
);

-- One row per matching run: which programmes were recommended to whom, when.
CREATE TABLE IF NOT EXISTS match_runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id    INTEGER NOT NULL REFERENCES clients(client_id) ON DELETE CASCADE,
    run_at       TEXT NOT NULL,
    top_k        INTEGER NOT NULL,
    program_ids  TEXT NOT NULL,                          -- JSON list of matched programme IDs
    llm_provider TEXT NOT NULL,
    report_path  TEXT NOT NULL
);
"""

_MUTABLE_CLIENT_FIELDS = {
    "company_name",
    "industry",
    "state",
    "company_size",
    "project_description",
}


# ---------------------------------------------------------------------------
# Connection / schema --------------------------------------------------------
# ---------------------------------------------------------------------------


def get_connection(settings: Settings) -> sqlite3.Connection:
    settings.ensure_directories()
    connection = sqlite3.connect(settings.sqlite_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def init_db(settings: Settings) -> None:
    """Create all tables if they do not exist (idempotent)."""
    with get_connection(settings) as connection:
        connection.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# Client CRUD ------------------------------------------------------------------
# ---------------------------------------------------------------------------


def add_client(
    settings: Settings,
    company_name: str,
    industry: str,
    state: str,
    project_description: str,
    company_size: str = "KMU",
) -> int:
    """Insert a client profile and return its new ``client_id``.

    ``state`` accepts canonical names and common abbreviations ("NRW",
    "Bayern", ...) and is stored in canonical form so it can be compared
    directly against the programmes' Fördergebiet metadata.
    """
    canonical_state = canonical_region(state)
    if canonical_state is None:
        raise ValueError(
            f"Unknown state {state!r}. Use a German Bundesland name, a common "
            f"abbreviation (e.g. 'NRW'), or 'bundesweit'."
        )
    if not project_description.strip():
        raise ValueError("project_description must not be empty (it drives the RAG matching).")

    now = utcnow_iso()
    with get_connection(settings) as connection:
        cursor = connection.execute(
            """
            INSERT INTO clients
                (company_name, industry, state, company_size, project_description,
                 created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                company_name.strip(),
                industry.strip(),
                canonical_state,
                company_size.strip(),
                project_description.strip(),
                now,
                now,
            ),
        )
        return int(cursor.lastrowid)


def get_client(settings: Settings, client_id: int) -> Optional[dict]:
    with get_connection(settings) as connection:
        row = connection.execute(
            "SELECT * FROM clients WHERE client_id = ?", (client_id,)
        ).fetchone()
    return dict(row) if row else None


def find_client_by_name(settings: Settings, company_name: str) -> Optional[dict]:
    with get_connection(settings) as connection:
        row = connection.execute(
            "SELECT * FROM clients WHERE company_name = ? ORDER BY client_id LIMIT 1",
            (company_name.strip(),),
        ).fetchone()
    return dict(row) if row else None


def list_clients(settings: Settings) -> list[dict]:
    with get_connection(settings) as connection:
        rows = connection.execute("SELECT * FROM clients ORDER BY client_id").fetchall()
    return [dict(row) for row in rows]


def update_client(settings: Settings, client_id: int, **fields: Any) -> bool:
    """Update selected client fields; returns ``True`` if a row was changed."""
    unknown = set(fields) - _MUTABLE_CLIENT_FIELDS
    if unknown:
        raise ValueError(f"Unknown client fields: {sorted(unknown)}")
    if not fields:
        return False
    if "state" in fields:
        canonical_state = canonical_region(str(fields["state"]))
        if canonical_state is None:
            raise ValueError(f"Unknown state {fields['state']!r}.")
        fields["state"] = canonical_state

    assignments = ", ".join(f"{name} = ?" for name in fields)
    values = list(fields.values()) + [utcnow_iso(), client_id]
    with get_connection(settings) as connection:
        cursor = connection.execute(
            f"UPDATE clients SET {assignments}, updated_at = ? WHERE client_id = ?",
            values,
        )
        return cursor.rowcount > 0


def delete_client(settings: Settings, client_id: int) -> bool:
    with get_connection(settings) as connection:
        cursor = connection.execute("DELETE FROM clients WHERE client_id = ?", (client_id,))
        return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Run logs --------------------------------------------------------------------
# ---------------------------------------------------------------------------


def log_ingestion_run(settings: Settings, source: str, stats: dict) -> None:
    with get_connection(settings) as connection:
        connection.execute(
            """
            INSERT INTO ingestion_runs (run_at, source, n_total, n_new, n_updated,
                                        n_unchanged, n_errors)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utcnow_iso(),
                source,
                stats.get("total", 0),
                stats.get("new", 0),
                stats.get("updated", 0),
                stats.get("unchanged", 0),
                stats.get("errors", 0),
            ),
        )


def log_match_run(
    settings: Settings,
    client_id: int,
    top_k: int,
    program_ids: list[str],
    llm_provider: str,
    report_path: str,
) -> None:
    with get_connection(settings) as connection:
        connection.execute(
            """
            INSERT INTO match_runs (client_id, run_at, top_k, program_ids,
                                    llm_provider, report_path)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (client_id, utcnow_iso(), top_k, json.dumps(program_ids), llm_provider, report_path),
        )
