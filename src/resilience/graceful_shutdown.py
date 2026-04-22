"""
src.resilience.graceful_shutdown — POSIX signal handling for clean exits.

Handles:
  SIGTERM / SIGINT  → set shutdown flag, run registered callbacks, exit cleanly
  SIGHUP            → reload config from disk without restarting

Signal handlers are installed at the OS level and are safe to call from any
thread.  Callbacks run in the main thread after the signal is received.

Usage
-----
    from src.resilience.graceful_shutdown import ShutdownHandler

    handler = ShutdownHandler()
    handler.on_shutdown(lambda: checkpoint_mgr.save_checkpoint(..., force=True))
    handler.register()

    # In main loop:
    while not handler.is_shutdown_requested():
        run_cycle()

    # After loop:
    handler.run_callbacks()  # or callbacks run automatically on signal
"""

from __future__ import annotations

import logging
import signal
import threading
from typing import Callable

logger = logging.getLogger(__name__)


class ShutdownHandler:
    """
    POSIX signal handler that coordinates graceful shutdowns and config reloads.

    Thread safety
    -------------
    The shutdown flag is set atomically via threading.Event.
    Callbacks are registered before signals arrive and called once, in order.
    All signal handler functions are safe to call from any thread (they only
    set the event and log; they do NOT call arbitrary Python code directly,
    which could deadlock in a signal context).

    Parameters
    ----------
    config_path : str | None
        Path to the settings file.  Used by the SIGHUP reload handler.
    exit_on_shutdown : bool
        If True, call sys.exit(0) after running callbacks on SIGTERM/SIGINT.
        Set to False in tests or when you handle the exit loop yourself.
    """

    def __init__(
        self,
        config_path: str | None = None,
        exit_on_shutdown: bool = False,
    ) -> None:
        self._shutdown_event = threading.Event()
        self._callbacks: list[Callable[[], None]] = []
        self._callbacks_lock = threading.Lock()
        self._config_path = config_path
        self._exit_on_shutdown = exit_on_shutdown
        self._registered = False
        self._callbacks_executed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self) -> None:
        """
        Install OS-level signal handlers for SIGTERM, SIGINT, and SIGHUP.

        Safe to call from the main thread only (signal.signal() requirement).
        Idempotent — calling more than once is a no-op.
        """
        if self._registered:
            logger.debug("ShutdownHandler: already registered")
            return

        signal.signal(signal.SIGTERM, self._handle_terminate)
        signal.signal(signal.SIGINT, self._handle_terminate)

        # SIGHUP is not available on Windows.
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self._handle_hup)

        self._registered = True
        logger.info(
            "ShutdownHandler: registered SIGTERM, SIGINT%s handlers",
            ", SIGHUP" if hasattr(signal, "SIGHUP") else "",
        )

    def is_shutdown_requested(self) -> bool:
        """
        Return True if a shutdown signal has been received.

        Call this inside your main loop:

            while not handler.is_shutdown_requested():
                run_cycle()
        """
        return self._shutdown_event.is_set()

    def on_shutdown(self, callback: Callable[[], None]) -> None:
        """
        Register a callback to be executed when a shutdown signal is received.

        Callbacks are called in registration order.
        If shutdown has already been requested (and the batch has already run),
        the callback is called immediately so late registrants are never dropped.

        Parameters
        ----------
        callback : callable
            Zero-argument callable (e.g. a lambda or bound method).
        """
        already_ran: bool
        with self._callbacks_lock:
            already_ran = self._callbacks_executed
            if not already_ran:
                self._callbacks.append(callback)

        if already_ran:
            # The batch has already executed — call this one directly.
            try:
                callback()
            except Exception as exc:
                logger.error("ShutdownHandler: late callback %r raised: %s", callback, exc)
        elif self._shutdown_event.is_set():
            # Shutdown requested but batch not yet executed — trigger now.
            self.run_callbacks()

    def run_callbacks(self) -> None:
        """
        Execute all registered shutdown callbacks, in order.

        Each callback's exceptions are caught and logged so that one failure
        does not prevent subsequent callbacks from running.

        Idempotent — callbacks run exactly once.
        """
        with self._callbacks_lock:
            if self._callbacks_executed:
                return
            self._callbacks_executed = True
            callbacks = list(self._callbacks)

        logger.info("ShutdownHandler: running %d shutdown callback(s)", len(callbacks))
        for cb in callbacks:
            try:
                cb()
            except Exception as exc:
                logger.error("ShutdownHandler: callback %r raised: %s", cb, exc)

    def request_shutdown(self) -> None:
        """
        Programmatically request a shutdown (as if SIGTERM was received).
        Useful for testing or internal abort logic.
        """
        logger.info("ShutdownHandler: shutdown requested programmatically")
        self._shutdown_event.set()
        self.run_callbacks()

    # ------------------------------------------------------------------
    # Signal handlers (called by the OS — must be signal-safe)
    # ------------------------------------------------------------------

    def _handle_terminate(self, signum: int, frame: object) -> None:
        """Handler for SIGTERM and SIGINT."""
        sig_name = signal.Signals(signum).name  # type: ignore[attr-defined]
        logger.warning("ShutdownHandler: received %s — requesting shutdown", sig_name)
        self._shutdown_event.set()
        # Schedule callback execution in a separate thread to avoid signal-handler
        # re-entrancy issues.
        t = threading.Thread(
            target=self._deferred_callbacks,
            name="shutdown-callbacks",
            daemon=True,
        )
        t.start()

    def _handle_hup(self, signum: int, frame: object) -> None:
        """Handler for SIGHUP — reload configuration from disk."""
        logger.info("ShutdownHandler: received SIGHUP — triggering config reload")
        t = threading.Thread(
            target=self._reload_config,
            name="sighup-config-reload",
            daemon=True,
        )
        t.start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _deferred_callbacks(self) -> None:
        """Run callbacks in a thread (called from signal handler thread)."""
        self.run_callbacks()
        if self._exit_on_shutdown:
            import sys
            logger.info("ShutdownHandler: exiting process")
            sys.exit(0)

    def _reload_config(self) -> None:
        """Reload the settings file from disk (SIGHUP handler)."""
        if self._config_path is None:
            logger.warning(
                "ShutdownHandler: SIGHUP received but no config_path set — ignoring."
            )
            return
        try:
            from config.settings import load_settings  # type: ignore[import]
            settings = load_settings(self._config_path)
            logger.info(
                "ShutdownHandler: config reloaded from %s (log_level=%s)",
                self._config_path,
                settings.log_level,
            )
        except Exception as exc:
            logger.error("ShutdownHandler: config reload failed: %s", exc)
