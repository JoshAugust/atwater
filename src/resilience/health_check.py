"""
src.resilience.health_check — System health verification for Atwater.

Usage
-----
    from src.resilience.health_check import HealthChecker

    checker = HealthChecker(lm_studio_url="http://localhost:1234/v1")
    results = checker.check_all()
    for r in results:
        status = "✓" if r.healthy else "✗"
        print(f"{status} {r.component}: {r.message} ({r.latency_ms:.1f}ms)")

Individual checks:
    r = checker.check_lm_studio()
    r = checker.check_databases(["data/state.db", "data/knowledge.db"])
    r = checker.check_embeddings()
    r = checker.check_disk_space(min_mb=200)
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class HealthResult:
    """
    Result of a single health check.

    Attributes
    ----------
    component : str
        Short name of the checked component (e.g. "lm_studio", "disk").
    healthy : bool
        True if the component is functioning correctly.
    message : str
        Human-readable explanation.
    latency_ms : float
        Time taken for the check in milliseconds.
    """

    component: str
    healthy: bool
    message: str
    latency_ms: float = 0.0

    def __str__(self) -> str:
        icon = "✓" if self.healthy else "✗"
        return f"[{icon}] {self.component}: {self.message} ({self.latency_ms:.1f}ms)"


# ---------------------------------------------------------------------------
# HealthChecker
# ---------------------------------------------------------------------------

class HealthChecker:
    """
    Orchestrates health checks across all Atwater subsystems.

    Parameters
    ----------
    lm_studio_url : str
        Base URL for the LM Studio API (e.g. "http://localhost:1234/v1").
    timeout : float
        HTTP timeout in seconds for network checks.
    """

    def __init__(
        self,
        lm_studio_url: str = "http://localhost:1234/v1",
        timeout: float = 5.0,
    ) -> None:
        self._lm_studio_url = lm_studio_url.rstrip("/")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def check_lm_studio(self, url: str | None = None) -> HealthResult:
        """
        Ping ``GET /v1/models`` to verify LM Studio is reachable and has a
        model loaded.

        Parameters
        ----------
        url : str | None
            Override the instance URL for this check.
        """
        base = (url or self._lm_studio_url).rstrip("/")
        endpoint = f"{base}/models"
        start = time.perf_counter()
        try:
            req = urllib.request.Request(endpoint, method="GET")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                elapsed_ms = (time.perf_counter() - start) * 1000

            data = json.loads(raw)
            models = [m.get("id", "?") for m in data.get("data", [])]
            if models:
                return HealthResult(
                    component="lm_studio",
                    healthy=True,
                    message=f"Reachable. {len(models)} model(s) loaded: {', '.join(models[:3])}",
                    latency_ms=elapsed_ms,
                )
            else:
                return HealthResult(
                    component="lm_studio",
                    healthy=False,
                    message="Server reachable but NO models loaded.",
                    latency_ms=elapsed_ms,
                )

        except urllib.error.URLError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return HealthResult(
                component="lm_studio",
                healthy=False,
                message=f"Connection failed: {exc}",
                latency_ms=elapsed_ms,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return HealthResult(
                component="lm_studio",
                healthy=False,
                message=f"Invalid response: {exc}",
                latency_ms=elapsed_ms,
            )

    def check_databases(
        self,
        paths: Sequence[str | Path],
    ) -> HealthResult:
        """
        Verify that all SQLite database files are readable by opening a
        connection and running ``PRAGMA integrity_check``.

        Parameters
        ----------
        paths : sequence of str or Path
            Paths to the database files to check.
        """
        start = time.perf_counter()
        issues: list[str] = []
        missing: list[str] = []

        for p in paths:
            path = Path(p)
            if not path.exists():
                missing.append(path.name)
                continue
            try:
                conn = sqlite3.connect(str(path), timeout=3.0)
                cursor = conn.execute("PRAGMA integrity_check")
                result = cursor.fetchone()
                conn.close()
                if result and result[0] != "ok":
                    issues.append(f"{path.name}: integrity={result[0]}")
            except sqlite3.Error as exc:
                issues.append(f"{path.name}: {exc}")

        elapsed_ms = (time.perf_counter() - start) * 1000

        if missing:
            # Missing DBs are acceptable (they'll be created on first run)
            msg_parts = [f"Missing (will create): {', '.join(missing)}"]
            if issues:
                msg_parts.append(f"Errors: {'; '.join(issues)}")
            return HealthResult(
                component="databases",
                healthy=len(issues) == 0,
                message=" | ".join(msg_parts),
                latency_ms=elapsed_ms,
            )

        if issues:
            return HealthResult(
                component="databases",
                healthy=False,
                message=f"Database errors: {'; '.join(issues)}",
                latency_ms=elapsed_ms,
            )

        checked = len(paths)
        return HealthResult(
            component="databases",
            healthy=True,
            message=f"All {checked} database(s) readable.",
            latency_ms=elapsed_ms,
        )

    def check_embeddings(self) -> HealthResult:
        """
        Attempt to import the sentence-transformers library and instantiate a
        small model as a smoke test.

        Does NOT download the model — only checks that the library is available
        and the import graph is intact.
        """
        start = time.perf_counter()
        try:
            import importlib
            importlib.import_module("sentence_transformers")
            elapsed_ms = (time.perf_counter() - start) * 1000
            return HealthResult(
                component="embeddings",
                healthy=True,
                message="sentence-transformers importable (model not loaded yet).",
                latency_ms=elapsed_ms,
            )
        except ImportError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return HealthResult(
                component="embeddings",
                healthy=False,
                message=f"sentence-transformers not available: {exc}. TF-IDF fallback will be used.",
                latency_ms=elapsed_ms,
            )

    def check_disk_space(self, min_mb: float = 100.0) -> HealthResult:
        """
        Verify that there is at least ``min_mb`` MB of free disk space.

        Parameters
        ----------
        min_mb : float
            Minimum required free space in megabytes.
        """
        start = time.perf_counter()
        try:
            usage = shutil.disk_usage(".")
            free_mb = usage.free / (1024 * 1024)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if free_mb >= min_mb:
                return HealthResult(
                    component="disk_space",
                    healthy=True,
                    message=f"{free_mb:.0f} MB free (required: {min_mb:.0f} MB).",
                    latency_ms=elapsed_ms,
                )
            else:
                return HealthResult(
                    component="disk_space",
                    healthy=False,
                    message=(
                        f"LOW DISK SPACE: {free_mb:.0f} MB free, "
                        f"need {min_mb:.0f} MB."
                    ),
                    latency_ms=elapsed_ms,
                )
        except OSError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return HealthResult(
                component="disk_space",
                healthy=False,
                message=f"Could not check disk space: {exc}",
                latency_ms=elapsed_ms,
            )

    def check_all(
        self,
        db_paths: Sequence[str | Path] | None = None,
        min_disk_mb: float = 100.0,
    ) -> list[HealthResult]:
        """
        Run all health checks and return results.

        Parameters
        ----------
        db_paths : sequence of paths, optional
            Database files to check.  Defaults to common Atwater paths.
        min_disk_mb : float
            Minimum free disk space in MB.
        """
        if db_paths is None:
            db_paths = ["data/state.db", "data/knowledge.db", "data/trials.db"]

        results: list[HealthResult] = [
            self.check_lm_studio(),
            self.check_databases(db_paths),
            self.check_embeddings(),
            self.check_disk_space(min_mb=min_disk_mb),
        ]
        return results

    def report(self, results: list[HealthResult] | None = None) -> str:
        """
        Format a multi-line health report string.

        If ``results`` is not provided, ``check_all()`` is called.
        """
        if results is None:
            results = self.check_all()

        lines = ["=== Atwater Health Check ==="]
        all_healthy = True
        for r in results:
            lines.append(f"  {r}")
            if not r.healthy:
                all_healthy = False

        status = "ALL SYSTEMS GO" if all_healthy else "⚠ ISSUES DETECTED"
        lines.append(f"\n  Overall: {status}")
        return "\n".join(lines)
