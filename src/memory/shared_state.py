"""
shared_state.py — SQLite-backed shared state machine.

Authoritative transient record of the current production run.
Supports concurrent agent access via WAL mode and role-based key scoping.

The role-scoping table (state_roles) maps each key to a set of agent roles
that are permitted to read it. The orchestrator always has read access to all
keys; other agents only see their filtered slice via state_read_scoped().

Schema
------
state (key TEXT PRIMARY KEY, value TEXT, updated_at REAL)
state_roles (key TEXT, role TEXT, PRIMARY KEY (key, role))

Atomic write pattern: Read → Calculate → Write inside a single transaction.
Use state_write() for individual atomic key updates. For compound
read-calculate-write logic, open a connection directly with context manager
and wrap in BEGIN IMMEDIATE.

Usage:
    sm = SharedState()
    sm.state_write("current_hypothesis", {"bg": "dark", "layout": "A"}, roles=["director", "creator"])
    value = sm.state_read("current_hypothesis")
    scoped = sm.state_read_scoped("creator")
    sm.state_write("score", 0.91, roles=["grader", "orchestrator"])
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

# Sentinel — all roles can read keys tagged with this token.
ROLE_ALL = "__all__"

# Default readable roles for the orchestrator (always has full access).
ORCHESTRATOR_ROLE = "orchestrator"


class SharedState:
    """
    SQLite-backed shared state machine with WAL mode and role-based scoping.

    Thread-safe for concurrent reads; writes are serialised by SQLite's
    WAL + busy_timeout. Each SharedState instance opens its own connection.
    """

    def __init__(self, db_path: str | Path = "state.db") -> None:
        """
        Initialise the shared state store.

        Args:
            db_path: Path to the SQLite database file. Defaults to ``state.db``
                     in the current working directory. Parent directories are
                     created automatically.
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._connect()
        self._bootstrap()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with WAL mode and busy_timeout configured."""
        conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            timeout=5.0,  # seconds — sqlite3 module-level timeout
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _bootstrap(self) -> None:
        """Create schema if not already present."""
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS state (
                    key        TEXT PRIMARY KEY,
                    value      TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS state_roles (
                    key   TEXT NOT NULL,
                    role  TEXT NOT NULL,
                    PRIMARY KEY (key, role),
                    FOREIGN KEY (key) REFERENCES state(key) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_state_roles_role ON state_roles(role);
                """
            )

    def _serialize(self, value: Any) -> str:
        """JSON-encode a Python value for storage."""
        return json.dumps(value, default=str)

    def _deserialize(self, raw: str) -> Any:
        """JSON-decode a stored value back to Python."""
        return json.loads(raw)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def state_read(self, key: str) -> Any:
        """
        Read a value from shared state by key (unrestricted).

        Intended for the orchestrator, which has access to all keys.

        Args:
            key: The state key to look up.

        Returns:
            The deserialized value, or ``None`` if the key does not exist.
        """
        row = self._conn.execute(
            "SELECT value FROM state WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return self._deserialize(row["value"])

    def state_write(
        self,
        key: str,
        value: Any,
        roles: list[str] | None = None,
    ) -> None:
        """
        Atomically write a value to shared state.

        The write is wrapped in a transaction so that the role tags and the
        value row are always consistent.

        Args:
            key: The state key to write.
            value: Any JSON-serialisable Python object.
            roles: List of agent role names that may read this key.
                   Pass ``[ROLE_ALL]`` (or omit) to grant access to all roles.
                   The orchestrator always has implicit read access regardless.
        """
        if roles is None:
            roles = [ROLE_ALL]

        serialized = self._serialize(value)
        now = time.time()

        with self._conn:
            # Upsert the value row.
            self._conn.execute(
                """
                INSERT INTO state (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                               updated_at = excluded.updated_at
                """,
                (key, serialized, now),
            )
            # Replace role assignments for this key.
            self._conn.execute("DELETE FROM state_roles WHERE key = ?", (key,))
            self._conn.executemany(
                "INSERT INTO state_roles (key, role) VALUES (?, ?)",
                [(key, role) for role in roles],
            )

    def state_read_scoped(self, agent_role: str) -> dict[str, Any]:
        """
        Return a filtered dict of all state keys readable by the given agent role.

        Keys tagged with ``ROLE_ALL`` are always included. The orchestrator role
        always receives all keys.

        Args:
            agent_role: The role name of the requesting agent
                        (e.g. ``"creator"``, ``"grader"``).

        Returns:
            A dict mapping key → deserialized value for all keys this role may
            read.
        """
        if agent_role == ORCHESTRATOR_ROLE:
            # Orchestrator sees everything.
            rows = self._conn.execute(
                "SELECT key, value FROM state"
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT s.key, s.value
                FROM state s
                JOIN state_roles sr ON sr.key = s.key
                WHERE sr.role = ? OR sr.role = ?
                GROUP BY s.key
                """,
                (agent_role, ROLE_ALL),
            ).fetchall()

        return {row["key"]: self._deserialize(row["value"]) for row in rows}

    def state_keys(self) -> list[str]:
        """Return all keys currently stored in shared state."""
        rows = self._conn.execute("SELECT key FROM state ORDER BY key").fetchall()
        return [row["key"] for row in rows]

    def state_delete(self, key: str) -> None:
        """
        Remove a key from shared state.

        Args:
            key: The state key to delete.
        """
        with self._conn:
            self._conn.execute("DELETE FROM state WHERE key = ?", (key,))

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()

    def __repr__(self) -> str:
        keys = self.state_keys()
        return f"SharedState(db={self._db_path!r}, keys={len(keys)})"
