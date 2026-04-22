"""
src.monitoring — Monitoring & Observability for the Atwater Cognitive Engine.

Provides structured JSONL event logging, real-time metrics collection, and a
live TUI dashboard (with Rich fallback when Textual is not installed).

Public API
----------
    from src.monitoring import AtwaterLogger, MetricsCollector, MetricsSummary
    from src.monitoring import AtwaterDashboard          # TUI (Textual or Rich)

Usage
-----
    logger = AtwaterLogger(log_dir="logs/", session_id="my-run")
    metrics = MetricsCollector()

    logger.log_event("cycle_start", {"cycle": 1})
    metrics.track_cycle(1, score=0.82, params={"style": "bold"}, duration_ms=3200)

    summary = metrics.get_summary()

    # Run the live dashboard (blocks until quit)
    dashboard = AtwaterDashboard(metrics_collector=metrics, logger=logger)
    dashboard.run()
"""

from .logger import AtwaterLogger
from .metrics import MetricsCollector, MetricsSummary

# Dashboard import is deferred/lazy because Textual & Rich are optional.
# Import explicitly: from src.monitoring.dashboard import AtwaterDashboard

__all__ = [
    "AtwaterLogger",
    "MetricsCollector",
    "MetricsSummary",
]
