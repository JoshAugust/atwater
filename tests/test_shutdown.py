"""
tests/test_shutdown.py — Tests for src.resilience.graceful_shutdown.

All tests run without sending real OS signals where possible.
Real SIGTERM/SIGINT sending is tested via os.kill(os.getpid(), ...) for
integration coverage.
"""

from __future__ import annotations

import os
import signal
import threading
import time

import pytest

from src.resilience.graceful_shutdown import ShutdownHandler


# ---------------------------------------------------------------------------
# Basic state
# ---------------------------------------------------------------------------

class TestBasicState:
    def test_not_requested_initially(self):
        h = ShutdownHandler()
        assert h.is_shutdown_requested() is False

    def test_request_shutdown_sets_flag(self):
        h = ShutdownHandler()
        h.request_shutdown()
        assert h.is_shutdown_requested() is True

    def test_multiple_request_shutdown_is_idempotent(self):
        h = ShutdownHandler()
        h.request_shutdown()
        h.request_shutdown()
        assert h.is_shutdown_requested() is True


# ---------------------------------------------------------------------------
# Callback registration and execution
# ---------------------------------------------------------------------------

class TestCallbacks:
    def test_callback_called_on_shutdown(self):
        h = ShutdownHandler()
        called = []
        h.on_shutdown(lambda: called.append(1))
        h.request_shutdown()
        assert called == [1]

    def test_multiple_callbacks_all_called(self):
        h = ShutdownHandler()
        log = []
        h.on_shutdown(lambda: log.append("a"))
        h.on_shutdown(lambda: log.append("b"))
        h.on_shutdown(lambda: log.append("c"))
        h.request_shutdown()
        assert log == ["a", "b", "c"]

    def test_callbacks_called_in_order(self):
        h = ShutdownHandler()
        order = []
        for i in range(5):
            i_copy = i
            h.on_shutdown(lambda x=i_copy: order.append(x))
        h.request_shutdown()
        assert order == list(range(5))

    def test_failing_callback_does_not_stop_others(self):
        h = ShutdownHandler()
        log = []
        h.on_shutdown(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        h.on_shutdown(lambda: log.append("survived"))
        h.request_shutdown()
        assert "survived" in log

    def test_callbacks_run_only_once(self):
        h = ShutdownHandler()
        count = []
        h.on_shutdown(lambda: count.append(1))
        h.request_shutdown()
        h.run_callbacks()  # second call should be no-op
        h.run_callbacks()
        assert len(count) == 1

    def test_late_callback_called_immediately_if_already_shutdown(self):
        h = ShutdownHandler()
        h.request_shutdown()
        called = []
        h.on_shutdown(lambda: called.append(1))
        # Should be called right away since shutdown already requested
        assert called == [1]


# ---------------------------------------------------------------------------
# Signal registration
# ---------------------------------------------------------------------------

class TestSignalRegistration:
    def test_register_is_idempotent(self):
        h = ShutdownHandler()
        h.register()
        h.register()  # should not raise
        # Restore default handlers so we don't pollute other tests
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    def test_register_installs_sigterm(self):
        h = ShutdownHandler()
        h.register()
        handler = signal.getsignal(signal.SIGTERM)
        assert handler == h._handle_terminate
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    def test_register_installs_sigint(self):
        h = ShutdownHandler()
        h.register()
        handler = signal.getsignal(signal.SIGINT)
        assert handler == h._handle_terminate
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    @pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="SIGHUP not available")
    def test_register_installs_sighup(self):
        h = ShutdownHandler()
        h.register()
        handler = signal.getsignal(signal.SIGHUP)
        assert handler == h._handle_hup
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)


# ---------------------------------------------------------------------------
# SIGTERM / SIGINT integration
# ---------------------------------------------------------------------------

class TestSignalIntegration:
    @pytest.mark.skipif(os.name != "posix", reason="POSIX only")
    def test_sigterm_sets_shutdown_flag(self):
        """Send SIGTERM to self and verify the shutdown flag is set."""
        h = ShutdownHandler(exit_on_shutdown=False)
        h.register()
        called = threading.Event()
        h.on_shutdown(lambda: called.set())

        os.kill(os.getpid(), signal.SIGTERM)

        # Give the deferred thread time to run
        assert called.wait(timeout=2.0), "Shutdown callback was not called within 2s"
        assert h.is_shutdown_requested()

        # Restore defaults
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)

    @pytest.mark.skipif(
        not hasattr(signal, "SIGHUP") or os.name != "posix",
        reason="SIGHUP not available or not POSIX"
    )
    def test_sighup_does_not_set_shutdown_flag(self):
        """SIGHUP should trigger config reload, NOT set the shutdown flag."""
        h = ShutdownHandler(exit_on_shutdown=False)
        h.register()

        os.kill(os.getpid(), signal.SIGHUP)
        time.sleep(0.1)  # give reload thread time to run

        assert not h.is_shutdown_requested()

        signal.signal(signal.SIGHUP, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_shutdowns_are_safe(self):
        """Multiple threads calling request_shutdown() concurrently is safe."""
        h = ShutdownHandler()
        count = []
        h.on_shutdown(lambda: count.append(1))

        threads = [threading.Thread(target=h.request_shutdown) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert h.is_shutdown_requested()
        # Callbacks should run exactly once
        assert len(count) == 1

    def test_concurrent_callback_registration(self):
        """Registering callbacks from multiple threads is safe."""
        h = ShutdownHandler()
        results = []

        def register_and_shutdown():
            h.on_shutdown(lambda: results.append(threading.current_thread().name))

        threads = [
            threading.Thread(target=register_and_shutdown, name=f"t{i}")
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        h.request_shutdown()
        # All registered callbacks should have run
        assert len(results) == 5
