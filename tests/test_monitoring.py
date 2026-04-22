"""
tests/test_monitoring.py — Tests for AtwaterLogger and MetricsCollector.

Run with:
    pytest tests/test_monitoring.py -v
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from src.monitoring.logger import AtwaterLogger, VALID_EVENT_TYPES
from src.monitoring.metrics import MetricsCollector, MetricsSummary


# ===========================================================================
# AtwaterLogger tests
# ===========================================================================


class TestAtwaterLogger:

    def test_creates_with_auto_session_id(self, tmp_path):
        logger = AtwaterLogger(log_dir=tmp_path)
        assert logger.session_id is not None
        assert len(logger.session_id) == 8  # 8 hex chars
        logger.close()

    def test_accepts_explicit_session_id(self, tmp_path):
        logger = AtwaterLogger(log_dir=tmp_path, session_id="test1234")
        assert logger.session_id == "test1234"
        logger.close()

    def test_log_event_writes_valid_jsonl(self, tmp_path):
        logger = AtwaterLogger(log_dir=tmp_path, session_id="testrun")
        logger.log_event("cycle_start", {"cycle": 1, "params": {"style": "bold"}})
        logger.flush()
        logger.close()

        # Find the log file
        log_files = list(tmp_path.glob("*.jsonl"))
        assert len(log_files) == 1, f"Expected 1 log file, got {len(log_files)}"

        lines = log_files[0].read_text().strip().splitlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["session_id"] == "testrun"
        assert record["event_type"] == "cycle_start"
        assert record["data"]["cycle"] == 1
        assert record["data"]["params"]["style"] == "bold"
        assert "ts" in record
        assert record["ts"].endswith("Z")

    def test_log_event_writes_multiple_events(self, tmp_path):
        logger = AtwaterLogger(log_dir=tmp_path)
        for i in range(5):
            logger.log_event("cycle_end", {"cycle": i, "score": 0.5 + i * 0.1})
        logger.flush()
        logger.close()

        log_files = list(tmp_path.glob("*.jsonl"))
        assert len(log_files) == 1

        lines = log_files[0].read_text().strip().splitlines()
        assert len(lines) == 5

        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["event_type"] == "cycle_end"
            assert abs(record["data"]["score"] - (0.5 + i * 0.1)) < 1e-9

    def test_cycle_number_embedded_in_events(self, tmp_path):
        logger = AtwaterLogger(log_dir=tmp_path)
        logger.set_cycle(7)
        logger.log_event("agent_call", {"role": "director"})
        logger.flush()
        logger.close()

        log_files = list(tmp_path.glob("*.jsonl"))
        record = json.loads(log_files[0].read_text().strip())
        assert record["cycle"] == 7

    def test_cycle_number_override(self, tmp_path):
        logger = AtwaterLogger(log_dir=tmp_path)
        logger.set_cycle(3)
        logger.log_event("error", {"role": "grader", "message": "oops"}, cycle_number=99)
        logger.flush()
        logger.close()

        log_files = list(tmp_path.glob("*.jsonl"))
        record = json.loads(log_files[0].read_text().strip())
        assert record["cycle"] == 99  # override wins

    def test_all_valid_event_types_accepted(self, tmp_path):
        logger = AtwaterLogger(log_dir=tmp_path)
        for et in VALID_EVENT_TYPES:
            logger.log_event(et, {})
        logger.flush()
        logger.close()

        log_files = list(tmp_path.glob("*.jsonl"))
        lines = log_files[0].read_text().strip().splitlines()
        assert len(lines) == len(VALID_EVENT_TYPES)

    def test_unknown_event_type_flagged(self, tmp_path):
        logger = AtwaterLogger(log_dir=tmp_path)
        logger.log_event("completely_made_up", {"foo": "bar"})
        logger.flush()
        logger.close()

        log_files = list(tmp_path.glob("*.jsonl"))
        record = json.loads(log_files[0].read_text().strip())
        assert record["_unknown_event_type"] is True

    def test_no_log_dir_does_not_crash(self):
        """Logger with log_dir=None must never crash."""
        logger = AtwaterLogger(log_dir=None)
        logger.log_event("cycle_start", {"cycle": 1})
        logger.flush()
        logger.close()

    def test_close_safe_to_call_multiple_times(self, tmp_path):
        logger = AtwaterLogger(log_dir=tmp_path)
        logger.log_event("checkpoint", {"cycle": 1, "path": "/tmp/cp.pkl"})
        logger.close()
        logger.close()  # should not raise

    def test_events_after_close_silently_dropped(self, tmp_path):
        logger = AtwaterLogger(log_dir=tmp_path)
        logger.log_event("cycle_start", {"cycle": 1})
        logger.flush()
        logger.close()
        logger.log_event("cycle_end", {"cycle": 1})  # should not raise

        log_files = list(tmp_path.glob("*.jsonl"))
        lines = log_files[0].read_text().strip().splitlines()
        assert len(lines) == 1  # second event not written

    def test_context_manager_support(self, tmp_path):
        with AtwaterLogger(log_dir=tmp_path) as logger:
            logger.log_event("cycle_start", {"cycle": 1})

        log_files = list(tmp_path.glob("*.jsonl"))
        assert len(log_files) == 1

    # --- Convenience helpers ---

    def test_convenience_cycle_start(self, tmp_path):
        with AtwaterLogger(log_dir=tmp_path) as logger:
            logger.cycle_start(1, {"style": "minimal"})

        log_files = list(tmp_path.glob("*.jsonl"))
        record = json.loads(log_files[0].read_text().strip())
        assert record["event_type"] == "cycle_start"
        assert record["cycle"] == 1

    def test_convenience_agent_result(self, tmp_path):
        with AtwaterLogger(log_dir=tmp_path) as logger:
            logger.agent_result("director", True, 840.0, tokens_in=1200, tokens_out=300)

        log_files = list(tmp_path.glob("*.jsonl"))
        record = json.loads(log_files[0].read_text().strip())
        assert record["event_type"] == "agent_result"
        assert record["data"]["role"] == "director"
        assert record["data"]["tokens_in"] == 1200

    def test_convenience_cascade_result(self, tmp_path):
        with AtwaterLogger(log_dir=tmp_path) as logger:
            logger.cascade_result(["fast", "medium"], short_circuited=True, total_ms=145.0)

        log_files = list(tmp_path.glob("*.jsonl"))
        record = json.loads(log_files[0].read_text().strip())
        assert record["event_type"] == "cascade_result"
        assert record["data"]["short_circuited"] is True

    # --- read_log ---

    def test_read_log_yields_all_events(self, tmp_path):
        with AtwaterLogger(log_dir=tmp_path) as logger:
            for i in range(10):
                logger.log_event("cycle_end", {"cycle": i})

        log_files = list(tmp_path.glob("*.jsonl"))
        events = list(AtwaterLogger.read_log(log_files[0]))
        assert len(events) == 10

    def test_read_log_skips_malformed_lines(self, tmp_path):
        log_file = tmp_path / "bad.jsonl"
        log_file.write_text(
            '{"event_type": "cycle_start", "ts": "2026-01-01T00:00:00Z", "session_id": "abc", "cycle": 1, "data": {}}\n'
            "this is not json\n"
            '{"event_type": "cycle_end", "ts": "2026-01-01T00:01:00Z", "session_id": "abc", "cycle": 1, "data": {}}\n'
        )
        events = list(AtwaterLogger.read_log(log_file))
        assert len(events) == 2

    def test_read_logs_multi_file(self, tmp_path):
        for i in range(3):
            p = tmp_path / f"run_{i}.jsonl"
            p.write_text(
                f'{{"event_type": "cycle_start", "ts": "2026-01-0{i+1}T00:00:00Z", '
                f'"session_id": "s{i}", "cycle": {i}, "data": {{}}}}\n'
            )

        events = list(AtwaterLogger.read_logs(list(tmp_path.glob("*.jsonl"))))
        assert len(events) == 3


# ===========================================================================
# MetricsCollector tests
# ===========================================================================


class TestMetricsCollector:

    def test_initial_summary_is_empty(self):
        m = MetricsCollector()
        s = m.get_summary()
        assert s.total_cycles == 0
        assert s.best_score is None
        assert s.avg_score is None
        assert s.score_trend == []
        assert s.total_tokens == {"in": 0, "out": 0, "total": 0}
        assert s.cascade_efficiency["total_evaluated"] == 0

    def test_track_cycle_basic(self):
        m = MetricsCollector()
        m.track_cycle(1, score=0.82, params={"style": "bold"}, duration_ms=3200)
        s = m.get_summary()
        assert s.total_cycles == 1
        assert abs(s.best_score - 0.82) < 1e-9
        assert abs(s.avg_score - 0.82) < 1e-9
        assert s.score_trend == [0.82]
        assert s.current_cycle == 1
        assert s.current_score == 0.82
        assert s.current_params == {"style": "bold"}

    def test_track_cycle_none_score(self):
        m = MetricsCollector()
        m.track_cycle(1, score=None, params={}, duration_ms=1000)
        s = m.get_summary()
        assert s.total_cycles == 1
        assert s.best_score is None
        assert s.avg_score is None

    def test_track_multiple_cycles(self):
        m = MetricsCollector()
        scores = [0.5, 0.7, 0.6, 0.9, 0.8]
        for i, sc in enumerate(scores, 1):
            m.track_cycle(i, score=sc, params={}, duration_ms=1000)
        s = m.get_summary()
        assert s.total_cycles == 5
        assert abs(s.best_score - 0.9) < 1e-9
        assert abs(s.avg_score - sum(scores) / len(scores)) < 1e-9

    def test_score_trend_last_20(self):
        m = MetricsCollector()
        for i in range(25):
            m.track_cycle(i + 1, score=i / 24.0, params={}, duration_ms=100)
        s = m.get_summary()
        assert len(s.score_trend) == 20
        # Should be the LAST 20 values
        assert abs(s.score_trend[-1] - 1.0) < 1e-9

    def test_avg_cycle_time_ms(self):
        m = MetricsCollector()
        m.track_cycle(1, score=0.5, params={}, duration_ms=1000)
        m.track_cycle(2, score=0.6, params={}, duration_ms=3000)
        s = m.get_summary()
        assert abs(s.avg_cycle_time_ms - 2000) < 1e-9

    def test_track_agent(self):
        m = MetricsCollector()
        m.track_agent("director", duration_ms=500, tokens_in=1000, tokens_out=200)
        m.track_agent("grader", duration_ms=2000, tokens_in=2000, tokens_out=500)
        m.track_agent("director", duration_ms=600, tokens_in=1100, tokens_out=250)

        s = m.get_summary()
        assert "director" in s.agent_stats
        assert s.agent_stats["director"]["calls"] == 2
        assert s.agent_stats["director"]["tokens_in"] == 2100
        assert s.agent_stats["director"]["tokens_out"] == 450
        assert abs(s.agent_stats["director"]["avg_ms"] - 550) < 1e-9

        assert s.total_tokens["in"] == 4100
        assert s.total_tokens["out"] == 950
        assert s.total_tokens["total"] == 5050

    def test_track_knowledge_write(self):
        m = MetricsCollector()
        m.track_knowledge("write", "observation", "e001")
        m.track_knowledge("write", "observation", "e002")
        m.track_knowledge("write", "rule", "r001")
        m.track_knowledge("promote", "rule", "r001")

        s = m.get_summary()
        assert s.kb_size_by_tier.get("observation") == 2
        assert s.kb_size_by_tier.get("rule") == 2  # 1 write + 1 promote

    def test_track_cascade_basic(self):
        m = MetricsCollector()
        m.track_cascade(["fast", "medium"], short_circuited=True, total_ms=150)
        m.track_cascade(["fast", "medium", "llm"], short_circuited=False, total_ms=4200)
        m.track_cascade(["fast"], short_circuited=True, total_ms=10)

        s = m.get_summary()
        ce = s.cascade_efficiency
        assert ce["total_evaluated"] == 3
        assert abs(ce["short_circuit_rate"] - 2 / 3) < 1e-9
        assert abs(ce["fast_pass_rate"] - 1.0) < 1e-9
        assert abs(ce["llm_reach_rate"] - 1 / 3) < 1e-9
        assert abs(ce["avg_time_ms"] - (150 + 4200 + 10) / 3) < 0.1

    def test_reset_clears_all_state(self):
        m = MetricsCollector()
        m.track_cycle(1, score=0.9, params={"x": 1}, duration_ms=1000)
        m.track_agent("director", duration_ms=500, tokens_in=100, tokens_out=50)
        m.track_knowledge("write", "observation", "e001")

        m.reset()

        s = m.get_summary()
        assert s.total_cycles == 0
        assert s.best_score is None
        assert s.total_tokens == {"in": 0, "out": 0, "total": 0}
        assert s.kb_size_by_tier == {}
        assert s.agent_stats == {}

    def test_get_summary_returns_dataclass(self):
        m = MetricsCollector()
        s = m.get_summary()
        assert isinstance(s, MetricsSummary)

    def test_thread_safety(self):
        """Multiple threads can write concurrently without raising."""
        import threading

        m = MetricsCollector()
        errors: list[Exception] = []

        def worker(start_cycle: int) -> None:
            try:
                for i in range(20):
                    cycle = start_cycle + i
                    m.track_cycle(cycle, score=0.5, params={}, duration_ms=100)
                    m.track_agent("director", duration_ms=50, tokens_in=100, tokens_out=30)
                    m.track_cascade(["fast"], short_circuited=True, total_ms=10)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i * 20,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        s = m.get_summary()
        assert s.total_cycles == 100

    def test_repr(self):
        m = MetricsCollector()
        m.track_cycle(1, score=0.9, params={}, duration_ms=1000)
        r = repr(m)
        assert "MetricsCollector" in r
        assert "0.9" in r
