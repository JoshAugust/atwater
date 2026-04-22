"""
working.py — Ephemeral per-turn working memory.

Holds variables, immediate inputs, and current-turn state.
Cleared between turns. Nothing important should live here permanently.

Usage:
    mem = WorkingMemory()
    mem.write("current_score", 0.91)
    score = mem.read("current_score")
    snap = mem.snapshot()
    mem.clear()
"""

from typing import Any


class WorkingMemory:
    """
    Ephemeral per-turn working memory.

    Simple dict-based store scoped to the current agent turn.
    Cleared between turns via clear() — callers are responsible for
    calling clear() at turn boundaries.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def read(self, key: str, default: Any = None) -> Any:
        """
        Read a value from working memory.

        Args:
            key: The key to look up.
            default: Value returned if key does not exist.

        Returns:
            The stored value, or default if not found.
        """
        return self._store.get(key, default)

    def write(self, key: str, value: Any) -> None:
        """
        Write a value to working memory.

        Args:
            key: The key to store under.
            value: Any serialisable Python value.
        """
        self._store[key] = value

    def clear(self) -> None:
        """
        Clear all entries from working memory.

        Call at the end of each agent turn to prevent stale state
        leaking into the next turn.
        """
        self._store.clear()

    def snapshot(self) -> dict[str, Any]:
        """
        Return a shallow copy of the current memory store.

        Useful for logging or checkpointing before a clear().

        Returns:
            A dict copy of the current in-memory store.
        """
        return dict(self._store)

    def __repr__(self) -> str:
        return f"WorkingMemory({len(self._store)} keys)"
