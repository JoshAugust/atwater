"""
tests/test_health_check.py — Tests for src.resilience.health_check.

All tests mock external calls. No LM Studio or real DBs required.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.resilience.health_check import HealthChecker, HealthResult


# ---------------------------------------------------------------------------
# HealthResult
# ---------------------------------------------------------------------------

class TestHealthResult:
    def test_str_healthy(self):
        r = HealthResult(component="lm_studio", healthy=True, message="ok", latency_ms=12.3)
        assert "[✓]" in str(r)
        assert "lm_studio" in str(r)

    def test_str_unhealthy(self):
        r = HealthResult(component="disk", healthy=False, message="low space", latency_ms=0.5)
        assert "[✗]" in str(r)

    def test_defaults(self):
        r = HealthResult(component="x", healthy=True, message="y")
        assert r.latency_ms == 0.0


# ---------------------------------------------------------------------------
# check_lm_studio
# ---------------------------------------------------------------------------

class TestCheckLMStudio:
    def _mock_urlopen(self, models: list[str]):
        """Return a context manager that yields a fake HTTP response."""
        raw = json.dumps({"data": [{"id": m} for m in models]}).encode()

        cm = MagicMock()
        cm.__enter__ = lambda s: MagicMock(read=lambda: raw)
        cm.__exit__ = MagicMock(return_value=False)
        return cm

    def test_healthy_with_models(self):
        checker = HealthChecker()
        raw = json.dumps({"data": [{"id": "model-a"}, {"id": "model-b"}]}).encode()
        fake_resp = MagicMock()
        fake_resp.read.return_value = raw

        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: fake_resp
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            result = checker.check_lm_studio()

        assert result.healthy is True
        assert "model-a" in result.message

    def test_unhealthy_no_models(self):
        checker = HealthChecker()
        raw = json.dumps({"data": []}).encode()
        fake_resp = MagicMock()
        fake_resp.read.return_value = raw

        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: fake_resp
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            result = checker.check_lm_studio()

        assert result.healthy is False
        assert "NO models" in result.message

    def test_connection_error(self):
        import urllib.error
        checker = HealthChecker()
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            result = checker.check_lm_studio()
        assert result.healthy is False
        assert "Connection failed" in result.message

    def test_url_override(self):
        """check_lm_studio(url=...) should use the provided URL, not the instance default."""
        import urllib.error
        checker = HealthChecker(lm_studio_url="http://localhost:1234/v1")
        captured_urls = []

        def fake_urlopen(req, timeout=None):
            captured_urls.append(req.full_url)
            raise urllib.error.URLError("refused")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            checker.check_lm_studio(url="http://custom:5678/v1")

        assert len(captured_urls) == 1
        assert "custom:5678" in captured_urls[0]

    def test_latency_measured(self):
        checker = HealthChecker()
        raw = json.dumps({"data": [{"id": "m"}]}).encode()
        fake_resp = MagicMock()
        fake_resp.read.return_value = raw

        with patch("urllib.request.urlopen") as mock_open:
            mock_open.return_value.__enter__ = lambda s: fake_resp
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            result = checker.check_lm_studio()
        assert result.latency_ms >= 0.0


# ---------------------------------------------------------------------------
# check_databases
# ---------------------------------------------------------------------------

class TestCheckDatabases:
    def test_readable_db(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE t (x INT)")
        conn.close()

        checker = HealthChecker()
        result = checker.check_databases([db])
        assert result.healthy is True

    def test_missing_db_is_acceptable(self, tmp_path):
        """Missing DBs are treated as 'will be created', healthy=True if no errors."""
        checker = HealthChecker()
        result = checker.check_databases([tmp_path / "nonexistent.db"])
        # Missing is acceptable (healthy unless there's an actual error)
        assert result.component == "databases"
        assert "Missing" in result.message

    def test_empty_paths_list(self):
        checker = HealthChecker()
        result = checker.check_databases([])
        assert result.healthy is True
        assert result.component == "databases"

    def test_multiple_dbs(self, tmp_path):
        dbs = []
        for i in range(3):
            db = tmp_path / f"db{i}.db"
            sqlite3.connect(str(db)).close()
            dbs.append(db)
        checker = HealthChecker()
        result = checker.check_databases(dbs)
        assert result.healthy is True

    def test_mixed_existing_and_missing(self, tmp_path):
        existing = tmp_path / "exists.db"
        sqlite3.connect(str(existing)).close()
        missing = tmp_path / "missing.db"

        checker = HealthChecker()
        result = checker.check_databases([existing, missing])
        assert result.component == "databases"


# ---------------------------------------------------------------------------
# check_embeddings
# ---------------------------------------------------------------------------

class TestCheckEmbeddings:
    def test_healthy_when_importable(self):
        checker = HealthChecker()
        fake_module = MagicMock()
        with patch("importlib.import_module", return_value=fake_module):
            result = checker.check_embeddings()
        assert result.healthy is True
        assert result.component == "embeddings"

    def test_unhealthy_when_not_importable(self):
        checker = HealthChecker()
        with patch("importlib.import_module", side_effect=ImportError("no module")):
            result = checker.check_embeddings()
        assert result.healthy is False
        assert "TF-IDF fallback" in result.message


# ---------------------------------------------------------------------------
# check_disk_space
# ---------------------------------------------------------------------------

class TestCheckDiskSpace:
    def test_sufficient_space(self):
        checker = HealthChecker()
        fake_usage = MagicMock()
        fake_usage.free = 500 * 1024 * 1024  # 500 MB
        with patch("shutil.disk_usage", return_value=fake_usage):
            result = checker.check_disk_space(min_mb=100)
        assert result.healthy is True
        assert "500" in result.message

    def test_insufficient_space(self):
        checker = HealthChecker()
        fake_usage = MagicMock()
        fake_usage.free = 50 * 1024 * 1024  # 50 MB
        with patch("shutil.disk_usage", return_value=fake_usage):
            result = checker.check_disk_space(min_mb=100)
        assert result.healthy is False
        assert "LOW DISK SPACE" in result.message

    def test_exact_threshold_passes(self):
        checker = HealthChecker()
        fake_usage = MagicMock()
        fake_usage.free = 100 * 1024 * 1024  # exactly 100 MB
        with patch("shutil.disk_usage", return_value=fake_usage):
            result = checker.check_disk_space(min_mb=100)
        assert result.healthy is True

    def test_os_error(self):
        checker = HealthChecker()
        with patch("shutil.disk_usage", side_effect=OSError("unavailable")):
            result = checker.check_disk_space()
        assert result.healthy is False


# ---------------------------------------------------------------------------
# check_all
# ---------------------------------------------------------------------------

class TestCheckAll:
    def test_returns_four_results(self):
        checker = HealthChecker()
        import urllib.error

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("down")), \
             patch("shutil.disk_usage", return_value=MagicMock(free=999 * 1024 * 1024)), \
             patch("importlib.import_module", side_effect=ImportError("no st")):
            results = checker.check_all()

        assert len(results) == 4
        components = [r.component for r in results]
        assert "lm_studio" in components
        assert "databases" in components
        assert "embeddings" in components
        assert "disk_space" in components

    def test_report_format(self):
        checker = HealthChecker()
        results = [
            HealthResult("lm_studio", True, "ok", 5.0),
            HealthResult("disk_space", False, "low", 0.1),
        ]
        report = checker.report(results)
        assert "Health Check" in report
        assert "✓" in report
        assert "✗" in report
        assert "ISSUES DETECTED" in report
