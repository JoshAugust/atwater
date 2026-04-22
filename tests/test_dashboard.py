"""
tests/test_dashboard.py — Dashboard initialisation tests.

We test that the dashboard can be constructed without error under all backend
configurations. We do NOT test rendering or TUI output — those require a live
terminal and are out of scope for CI.

Run with:
    pytest tests/test_dashboard.py -v
"""

from __future__ import annotations

import sys

import pytest

from src.monitoring.metrics import MetricsCollector, MetricsSummary
from src.monitoring.logger import AtwaterLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_summary(**overrides) -> MetricsSummary:
    """Create a populated MetricsSummary for testing."""
    defaults = dict(
        total_cycles=5,
        best_score=0.82,
        avg_score=0.71,
        score_trend=[0.60, 0.65, 0.70, 0.78, 0.82],
        kb_size_by_tier={"observation": 10, "rule": 3},
        cascade_efficiency={
            "total_evaluated": 5,
            "short_circuit_rate": 0.4,
            "fast_pass_rate": 0.8,
            "medium_pass_rate": 0.6,
            "llm_reach_rate": 0.6,
            "avg_time_ms": 1200.0,
        },
        total_tokens={"in": 12000, "out": 3000, "total": 15000},
        avg_cycle_time_ms=3200.0,
        agent_stats={
            "director": {"calls": 5, "total_ms": 2500, "avg_ms": 500, "tokens_in": 5000, "tokens_out": 1000},
            "grader": {"calls": 5, "total_ms": 10000, "avg_ms": 2000, "tokens_in": 7000, "tokens_out": 2000},
        },
        current_cycle=5,
        current_params={"style": "bold", "palette": "warm"},
        current_score=0.82,
    )
    defaults.update(overrides)
    return MetricsSummary(**defaults)


# ---------------------------------------------------------------------------
# AtwaterDashboard: facade class
# ---------------------------------------------------------------------------


class TestAtwaterDashboardInit:

    def test_import_succeeds(self):
        from src.monitoring.dashboard import AtwaterDashboard
        assert AtwaterDashboard is not None

    def test_creates_with_metrics_only(self):
        from src.monitoring.dashboard import AtwaterDashboard

        mc = MetricsCollector()
        dashboard = AtwaterDashboard(metrics_collector=mc)
        assert dashboard is not None

    def test_creates_with_logger(self, tmp_path):
        from src.monitoring.dashboard import AtwaterDashboard

        mc = MetricsCollector()
        logger = AtwaterLogger(log_dir=tmp_path)
        dashboard = AtwaterDashboard(metrics_collector=mc, logger=logger)
        assert dashboard is not None
        logger.close()

    def test_backend_name_is_valid(self):
        from src.monitoring.dashboard import AtwaterDashboard

        mc = MetricsCollector()
        dashboard = AtwaterDashboard(metrics_collector=mc)
        assert dashboard.backend_name in ("textual", "rich", "plain")

    def test_force_backend_plain(self):
        from src.monitoring.dashboard import AtwaterDashboard

        mc = MetricsCollector()
        dashboard = AtwaterDashboard(metrics_collector=mc, force_backend="plain")
        # plain backend should always succeed regardless of installed packages
        assert dashboard.backend_name == "plain"
        assert dashboard._backend is not None

    def test_force_backend_rich_if_available(self):
        from src.monitoring.dashboard import AtwaterDashboard, _RICH_AVAILABLE

        mc = MetricsCollector()
        dashboard = AtwaterDashboard(metrics_collector=mc, force_backend="rich")
        # Dashboard is created; backend may gracefully degrade
        assert dashboard._backend is not None

    def test_update_does_not_crash(self):
        from src.monitoring.dashboard import AtwaterDashboard

        mc = MetricsCollector()
        dashboard = AtwaterDashboard(metrics_collector=mc, force_backend="plain")
        summary = make_summary()
        # Should not raise, even with plain backend
        dashboard.update(summary)

    def test_repr_includes_backend(self):
        from src.monitoring.dashboard import AtwaterDashboard

        mc = MetricsCollector()
        dashboard = AtwaterDashboard(metrics_collector=mc, force_backend="plain")
        r = repr(dashboard)
        assert "AtwaterDashboard" in r
        assert "plain" in r

    def test_refresh_interval_stored(self):
        from src.monitoring.dashboard import AtwaterDashboard

        mc = MetricsCollector()
        dashboard = AtwaterDashboard(metrics_collector=mc, refresh_interval=10.0)
        assert dashboard._refresh == 10.0

    def test_stop_does_not_crash(self):
        from src.monitoring.dashboard import AtwaterDashboard

        mc = MetricsCollector()
        dashboard = AtwaterDashboard(metrics_collector=mc, force_backend="plain")
        dashboard.stop()  # should not raise


# ---------------------------------------------------------------------------
# _RichDashboard (fallback) — unit tests
# ---------------------------------------------------------------------------


class TestRichDashboardFallback:

    def test_creates_without_error(self):
        from src.monitoring.dashboard import _RichDashboard

        mc = MetricsCollector()
        d = _RichDashboard(metrics_collector=mc, refresh_interval=5.0)
        assert d is not None

    def test_update_stores_summary(self):
        from src.monitoring.dashboard import _RichDashboard

        mc = MetricsCollector()
        d = _RichDashboard(metrics_collector=mc, refresh_interval=5.0)
        s = make_summary()
        d.update(s)
        assert d._latest_summary is s

    def test_pause_and_resume(self):
        from src.monitoring.dashboard import _RichDashboard

        mc = MetricsCollector()
        d = _RichDashboard(metrics_collector=mc, refresh_interval=5.0)
        assert not d._paused
        d.pause()
        assert d._paused
        d.resume()
        assert not d._paused

    def test_stop_sets_running_false(self):
        from src.monitoring.dashboard import _RichDashboard

        mc = MetricsCollector()
        d = _RichDashboard(metrics_collector=mc, refresh_interval=5.0)
        d._running = True
        d.stop()
        assert not d._running


# ---------------------------------------------------------------------------
# Helpers / sparkline
# ---------------------------------------------------------------------------


class TestHelpers:

    def test_sparkline_empty(self):
        from src.monitoring.dashboard import _sparkline

        result = _sparkline([], width=10)
        assert result == "─" * 10

    def test_sparkline_single_value(self):
        from src.monitoring.dashboard import _sparkline

        result = _sparkline([0.5], width=10)
        assert len(result) == 1  # single value → single char

    def test_sparkline_all_same(self):
        from src.monitoring.dashboard import _sparkline

        result = _sparkline([0.7, 0.7, 0.7], width=3)
        assert len(result) == 3

    def test_sparkline_ascending(self):
        from src.monitoring.dashboard import _sparkline

        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = _sparkline(values, width=5)
        assert len(result) == 5

    def test_cost_estimate_format(self):
        from src.monitoring.dashboard import _cost_estimate

        result = _cost_estimate(1000, 500)
        assert "$" in result
        assert "OpenAI" in result

    def test_fmt_ms_none(self):
        from src.monitoring.dashboard import _fmt_ms

        assert _fmt_ms(None) == "N/A"

    def test_fmt_ms_short(self):
        from src.monitoring.dashboard import _fmt_ms

        result = _fmt_ms(500)
        assert "ms" in result

    def test_fmt_ms_long(self):
        from src.monitoring.dashboard import _fmt_ms

        result = _fmt_ms(5000)
        assert "s" in result


# ---------------------------------------------------------------------------
# Textual availability check (informational, not a hard failure)
# ---------------------------------------------------------------------------


class TestTextualAvailability:

    def test_textual_flag_is_bool(self):
        from src.monitoring.dashboard import _TEXTUAL_AVAILABLE

        assert isinstance(_TEXTUAL_AVAILABLE, bool)

    def test_rich_flag_is_bool(self):
        from src.monitoring.dashboard import _RICH_AVAILABLE

        assert isinstance(_RICH_AVAILABLE, bool)

    @pytest.mark.skipif(
        True,  # Skip by default — requires textual to be installed
        reason="Only run when textual is installed: pip install textual>=0.80",
    )
    def test_textual_app_created_when_available(self):
        from src.monitoring.dashboard import _TEXTUAL_AVAILABLE, _build_textual_dashboard
        from src.monitoring.metrics import MetricsCollector

        if not _TEXTUAL_AVAILABLE:
            pytest.skip("Textual not installed")

        mc = MetricsCollector()
        app = _build_textual_dashboard(mc, None, 2.0)
        assert app is not None
