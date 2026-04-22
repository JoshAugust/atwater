"""
src.monitoring.dashboard — Live TUI dashboard for the Atwater Cognitive Engine.

Two rendering backends:
1. Textual (preferred) — full TUI with panels, keyboard handling, and live
   updates.  Requires ``textual>=0.80``.
2. Rich fallback — a compact status table printed to stdout every N seconds
   when Textual is not installed or when running non-interactively.

Keyboard shortcuts (Textual mode only)
---------------------------------------
    q   quit
    p   pause updates
    r   resume updates
    d   toggle detail view

Public API
----------
    from src.monitoring.dashboard import AtwaterDashboard

    dashboard = AtwaterDashboard(metrics_collector=mc, logger=logger)
    dashboard.run()                  # blocks until quit
    dashboard.update(summary)        # push a new summary from outside
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .logger import AtwaterLogger
    from .metrics import MetricsCollector, MetricsSummary

# ---------------------------------------------------------------------------
# Textual availability probe
# ---------------------------------------------------------------------------

_TEXTUAL_AVAILABLE = False
try:
    import textual  # noqa: F401
    _TEXTUAL_AVAILABLE = True
except ImportError:
    pass

_RICH_AVAILABLE = False
try:
    import rich  # noqa: F401
    _RICH_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sparkline(values: list[float], width: int = 30) -> str:
    """Render a list of 0–1 float scores as a Unicode sparkline."""
    if not values:
        return "─" * width
    blocks = " ▁▂▃▄▅▆▇█"
    sample = values[-width:]
    lo, hi = min(sample), max(sample)
    span = hi - lo if hi > lo else 1.0
    return "".join(blocks[int((v - lo) / span * 8)] for v in sample)


def _cost_estimate(tokens_in: int, tokens_out: int) -> str:
    """Rough cost estimate for LM Studio / local model (free), or OpenAI pricing."""
    # Local model: $0 — but show token count
    return f"~${(tokens_in * 0.000003 + tokens_out * 0.000015):.4f} (OpenAI equiv.)"


def _fmt_ms(ms: float | None) -> str:
    if ms is None:
        return "N/A"
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


# ===========================================================================
# Rich fallback dashboard
# ===========================================================================


class _RichDashboard:
    """
    Simple console dashboard using Rich tables.

    Prints a status snapshot every ``refresh_interval`` seconds.
    Runs in the foreground on the calling thread.
    """

    def __init__(
        self,
        metrics_collector: "MetricsCollector",
        logger: "AtwaterLogger | None" = None,
        refresh_interval: float = 5.0,
    ) -> None:
        self._metrics = metrics_collector
        self._logger = logger
        self._refresh = refresh_interval
        self._paused = False
        self._running = False
        self._latest_summary: "MetricsSummary | None" = None

    # ------------------------------------------------------------------
    # Public API mirrors AtwaterDashboard
    # ------------------------------------------------------------------

    def update(self, summary: "MetricsSummary") -> None:
        self._latest_summary = summary

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        """Block, printing status every refresh_interval seconds."""
        if not _RICH_AVAILABLE:
            self._run_plain()
            return
        self._run_rich()

    # ------------------------------------------------------------------
    # Plain (no Rich) fallback
    # ------------------------------------------------------------------

    def _run_plain(self) -> None:
        self._running = True
        start = time.time()
        print("[Atwater Dashboard] (plain mode — install rich or textual for TUI)")
        try:
            while self._running:
                if not self._paused:
                    s = self._latest_summary or self._metrics.get_summary()
                    elapsed = time.time() - start
                    print(
                        f"\n--- Atwater | Cycle {s.current_cycle} | "
                        f"Elapsed {elapsed:.0f}s ---"
                    )
                    print(f"  Best score  : {s.best_score:.4f}" if s.best_score is not None else "  Best score  : N/A")
                    print(f"  Avg score   : {s.avg_score:.4f}" if s.avg_score is not None else "  Avg score   : N/A")
                    print(f"  Total cycles: {s.total_cycles}")
                    print(f"  Tokens      : in={s.total_tokens['in']:,} out={s.total_tokens['out']:,}")
                    kb_str = ", ".join(f"{t}={c}" for t, c in s.kb_size_by_tier.items()) or "empty"
                    print(f"  Knowledge   : {kb_str}")
                    sc_rate = s.cascade_efficiency.get("short_circuit_rate", 0)
                    print(f"  Cascade SC% : {sc_rate:.0%}")
                time.sleep(self._refresh)
        except KeyboardInterrupt:
            pass

    # ------------------------------------------------------------------
    # Rich-powered fallback
    # ------------------------------------------------------------------

    def _run_rich(self) -> None:
        from rich.console import Console
        from rich.table import Table
        from rich.live import Live
        from rich.layout import Layout
        from rich.panel import Panel
        from rich.text import Text

        console = Console()
        self._running = True
        start = time.time()

        def _build_renderable() -> Any:
            s = self._latest_summary or self._metrics.get_summary()
            elapsed = time.time() - start
            h, m, sec = int(elapsed // 3600), int((elapsed % 3600) // 60), int(elapsed % 60)
            elapsed_str = f"{h:02d}:{m:02d}:{sec:02d}"

            # Title
            title_line = (
                f"[bold cyan]Atwater Cognitive Engine[/bold cyan]  "
                f"Cycle [bold]{s.current_cycle}[/bold]  "
                f"Elapsed [green]{elapsed_str}[/green]"
                + ("  [yellow][PAUSED][/yellow]" if self._paused else "")
            )

            # Score trend sparkline
            spark = _sparkline(s.score_trend, width=40)
            best_str = f"{s.best_score:.4f}" if s.best_score is not None else "N/A"
            avg_str = f"{s.avg_score:.4f}" if s.avg_score is not None else "N/A"
            score_panel = Panel(
                f"[dim]Last {len(s.score_trend)} scores[/dim]\n{spark}\n"
                f"Best: [green]{best_str}[/green]  Avg: [cyan]{avg_str}[/cyan]",
                title="Score Trend",
                border_style="green",
            )

            # Current cycle panel
            params_str = (
                "\n".join(f"  {k}: {v}" for k, v in s.current_params.items())
                or "  (none yet)"
            )
            score_now = f"{s.current_score:.4f}" if s.current_score is not None else "N/A"
            cycle_panel = Panel(
                f"Score: [bold]{score_now}[/bold]\n\nParams:\n{params_str}",
                title=f"Cycle {s.current_cycle}",
                border_style="blue",
            )

            # KB health
            kb_rows = "\n".join(
                f"  {tier:<14}: {count}"
                for tier, count in s.kb_size_by_tier.items()
            ) or "  (empty)"
            kb_panel = Panel(kb_rows, title="Knowledge Base", border_style="magenta")

            # Cascade
            ce = s.cascade_efficiency
            cascade_text = (
                f"  Evaluated    : {ce.get('total_evaluated', 0)}\n"
                f"  Short-circuit: {ce.get('short_circuit_rate', 0):.0%}\n"
                f"  Fast pass    : {ce.get('fast_pass_rate', 0):.0%}\n"
                f"  Medium pass  : {ce.get('medium_pass_rate', 0):.0%}\n"
                f"  LLM reach    : {ce.get('llm_reach_rate', 0):.0%}\n"
                f"  Avg time     : {_fmt_ms(ce.get('avg_time_ms'))}"
            )
            cascade_panel = Panel(cascade_text, title="Cascade Stats", border_style="yellow")

            # Token usage
            ti = s.total_tokens["in"]
            to = s.total_tokens["out"]
            tt = s.total_tokens["total"]
            token_text = (
                f"  In   : {ti:>10,}\n"
                f"  Out  : {to:>10,}\n"
                f"  Total: {tt:>10,}\n"
                f"  Cost : {_cost_estimate(ti, to)}"
            )
            token_panel = Panel(token_text, title="Token Usage", border_style="cyan")

            # Agents
            agent_rows = []
            for role, stats in s.agent_stats.items():
                agent_rows.append(
                    f"  {role:<16}: {stats['calls']} calls  "
                    f"avg={_fmt_ms(stats['avg_ms'])}  "
                    f"tok={stats['tokens_in']:,}+{stats['tokens_out']:,}"
                )
            agent_text = "\n".join(agent_rows) or "  (none yet)"
            agent_panel = Panel(agent_text, title="Agent Stats", border_style="white")

            from rich.columns import Columns

            body = (
                title_line + "\n\n"
            )
            return Panel(
                Columns(
                    [score_panel, cycle_panel, kb_panel, cascade_panel, token_panel, agent_panel],
                    equal=False,
                    expand=True,
                ),
                title="[bold]Atwater Cognitive Engine[/bold]",
                subtitle=f"Cycle {s.current_cycle} | {elapsed_str} | q=quit p=pause r=resume",
            )

        try:
            with Live(_build_renderable(), console=console, refresh_per_second=1) as live:
                while self._running:
                    time.sleep(self._refresh)
                    if not self._paused:
                        live.update(_build_renderable())
        except KeyboardInterrupt:
            pass


# ===========================================================================
# Textual dashboard (full TUI)
# ===========================================================================


def _build_textual_dashboard(
    metrics_collector: "MetricsCollector",
    logger: "AtwaterLogger | None",
    refresh_interval: float,
) -> Any:
    """Construct and return a Textual App instance."""
    from textual.app import App, ComposeResult
    from textual.widgets import Header, Footer, Static, Label
    from textual.reactive import reactive
    from textual.containers import Horizontal, Vertical
    from textual import work

    class ScoreTrendWidget(Static):
        scores: reactive[list[float]] = reactive([], recompose=True)

        def render(self) -> str:
            if not self.scores:
                return "[dim]No scores yet[/dim]"
            spark = _sparkline(self.scores, width=50)
            best = max(self.scores)
            avg = sum(self.scores) / len(self.scores)
            return (
                f"[dim]Last {len(self.scores)} scores[/dim]\n"
                f"{spark}\n"
                f"Best: [green]{best:.4f}[/green]  Avg: [cyan]{avg:.4f}[/cyan]"
            )

    class CycleInfoWidget(Static):
        cycle: reactive[int] = reactive(0)
        score: reactive[str] = reactive("N/A")
        params: reactive[str] = reactive("(none)")

        def render(self) -> str:
            return (
                f"[bold]Cycle {self.cycle}[/bold]  Score: [green]{self.score}[/green]\n"
                f"\nParams:\n{self.params}"
            )

    class KBHealthWidget(Static):
        tiers: reactive[str] = reactive("(empty)")

        def render(self) -> str:
            return f"[bold]Knowledge Base:[/bold]\n{self.tiers}"

    class OptunaWidget(Static):
        content: reactive[str] = reactive("(no trials yet)")

        def render(self) -> str:
            return f"[bold]Optuna Insights:[/bold]\n{self.content}"

    class CascadeWidget(Static):
        content: reactive[str] = reactive("(no evaluations yet)")

        def render(self) -> str:
            return f"[bold]Cascade Stats:[/bold]\n{self.content}"

    class TokenWidget(Static):
        content: reactive[str] = reactive("in=0 out=0")

        def render(self) -> str:
            return f"[bold]Token Usage:[/bold]\n{self.content}"

    class AlertsWidget(Static):
        content: reactive[str] = reactive("[green]No alerts[/green]")

        def render(self) -> str:
            return f"[bold]Alerts:[/bold]\n{self.content}"

    class AtwaterApp(App):
        """Atwater live TUI dashboard."""

        CSS = """
        Screen { background: #0d1117; }
        Header { background: #161b22; color: #58a6ff; }
        Footer { background: #161b22; }
        ScoreTrendWidget { height: 6; border: solid #30363d; padding: 0 1; }
        CycleInfoWidget  { height: 10; border: solid #30363d; padding: 0 1; }
        KBHealthWidget   { height: 8; border: solid #30363d; padding: 0 1; }
        OptunaWidget     { height: 8; border: solid #30363d; padding: 0 1; }
        CascadeWidget    { height: 10; border: solid #30363d; padding: 0 1; }
        TokenWidget      { height: 8; border: solid #30363d; padding: 0 1; }
        AlertsWidget     { height: 8; border: solid #e3b341; padding: 0 1; }
        """

        BINDINGS = [
            ("q", "quit", "Quit"),
            ("p", "pause", "Pause"),
            ("r", "resume", "Resume"),
            ("d", "detail", "Detail"),
        ]

        def __init__(self, mc: "MetricsCollector", lg: "AtwaterLogger | None", interval: float) -> None:
            super().__init__()
            self._mc = mc
            self._lg = lg
            self._interval = interval
            self._paused = False
            self._detail = False
            self._latest_summary: "MetricsSummary | None" = None

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Vertical():
                yield ScoreTrendWidget(id="score-trend")
                with Horizontal():
                    yield CycleInfoWidget(id="cycle-info")
                    yield KBHealthWidget(id="kb-health")
                with Horizontal():
                    yield OptunaWidget(id="optuna")
                    yield CascadeWidget(id="cascade")
                with Horizontal():
                    yield TokenWidget(id="tokens")
                    yield AlertsWidget(id="alerts")
            yield Footer()

        def on_mount(self) -> None:
            self.title = "Atwater Cognitive Engine"
            self._refresh_loop()

        @work(exclusive=True, thread=True)
        def _refresh_loop(self) -> None:
            import time as _time
            while True:
                _time.sleep(self._interval)
                if not self._paused:
                    self.call_from_thread(self._do_refresh)

        def _do_refresh(self) -> None:
            summary = self._latest_summary or self._mc.get_summary()
            self._apply_summary(summary)

        def _apply_summary(self, s: "MetricsSummary") -> None:
            # Score trend
            trend_w = self.query_one("#score-trend", ScoreTrendWidget)
            trend_w.scores = s.score_trend

            # Cycle info
            cycle_w = self.query_one("#cycle-info", CycleInfoWidget)
            cycle_w.cycle = s.current_cycle
            cycle_w.score = f"{s.current_score:.4f}" if s.current_score is not None else "N/A"
            cycle_w.params = "\n".join(f"  {k}: {v}" for k, v in s.current_params.items()) or "  (none)"

            # KB health
            kb_w = self.query_one("#kb-health", KBHealthWidget)
            kb_w.tiers = "\n".join(
                f"  {t}: {c}" for t, c in s.kb_size_by_tier.items()
            ) or "  (empty)"

            # Optuna (best params from summary.current_params)
            opt_w = self.query_one("#optuna", OptunaWidget)
            opt_w.content = "\n".join(
                f"  {k}: {v}" for k, v in s.current_params.items()
            ) or "  (no trials)"

            # Cascade
            ce = s.cascade_efficiency
            casc_w = self.query_one("#cascade", CascadeWidget)
            casc_w.content = (
                f"  Evaluated    : {ce.get('total_evaluated', 0)}\n"
                f"  Short-circuit: {ce.get('short_circuit_rate', 0):.0%}\n"
                f"  Fast pass    : {ce.get('fast_pass_rate', 0):.0%}\n"
                f"  LLM reach    : {ce.get('llm_reach_rate', 0):.0%}\n"
                f"  Avg time     : {_fmt_ms(ce.get('avg_time_ms'))}"
            )

            # Tokens
            ti, to, tt = s.total_tokens["in"], s.total_tokens["out"], s.total_tokens["total"]
            tok_w = self.query_one("#tokens", TokenWidget)
            tok_w.content = (
                f"  In   : {ti:>10,}\n"
                f"  Out  : {to:>10,}\n"
                f"  Total: {tt:>10,}\n"
                f"  Cost : {_cost_estimate(ti, to)}"
            )

            # Alerts (placeholder — wire in real circuit-breaker state)
            alert_w = self.query_one("#alerts", AlertsWidget)
            alert_w.content = "[green]No alerts[/green]"

        # Public update hook — call from the run loop
        def update(self, summary: "MetricsSummary") -> None:
            self._latest_summary = summary
            if not self._paused:
                self.call_later(self._apply_summary, summary)

        # Keybindings
        def action_pause(self) -> None:
            self._paused = True

        def action_resume(self) -> None:
            self._paused = False

        def action_detail(self) -> None:
            self._detail = not self._detail

    return AtwaterApp(mc=metrics_collector, lg=logger, interval=refresh_interval)


# ===========================================================================
# Public facade: AtwaterDashboard
# ===========================================================================


class AtwaterDashboard:
    """
    Live monitoring dashboard for the Atwater Cognitive Engine.

    Automatically selects the best available backend:
    - Textual TUI (if ``textual>=0.80`` is installed)
    - Rich live console (if ``rich`` is installed)
    - Plain stdout fallback otherwise

    Parameters
    ----------
    metrics_collector : MetricsCollector
        Shared metrics instance.
    logger : AtwaterLogger | None
        Optional logger for session context.
    refresh_interval : float
        Seconds between display refreshes (default 2.0).
    force_backend : str | None
        Force a specific backend: "textual", "rich", or "plain".
    """

    def __init__(
        self,
        metrics_collector: "MetricsCollector",
        logger: "AtwaterLogger | None" = None,
        refresh_interval: float = 2.0,
        force_backend: str | None = None,
    ) -> None:
        self._metrics = metrics_collector
        self._logger = logger
        self._refresh = refresh_interval
        self._backend_name = force_backend or self._detect_backend()
        self._backend: Any = self._build_backend()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        """Which backend is active: "textual", "rich", or "plain"."""
        return self._backend_name

    def update(self, summary: "MetricsSummary") -> None:
        """Push a new MetricsSummary to the dashboard."""
        if hasattr(self._backend, "update"):
            self._backend.update(summary)

    def run(self) -> None:
        """Start the dashboard.  Blocks until the user quits."""
        if self._backend_name == "textual":
            self._backend.run()
        else:
            self._backend.run()

    def stop(self) -> None:
        """Stop the dashboard loop."""
        if hasattr(self._backend, "stop"):
            self._backend.stop()
        if hasattr(self._backend, "exit"):
            self._backend.exit()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _detect_backend(self) -> str:
        if _TEXTUAL_AVAILABLE:
            return "textual"
        if _RICH_AVAILABLE:
            return "rich"
        return "plain"

    def _build_backend(self) -> Any:
        if self._backend_name == "textual":
            return _build_textual_dashboard(
                self._metrics, self._logger, self._refresh
            )
        # Both "rich" and "plain" use _RichDashboard (which handles both)
        return _RichDashboard(
            metrics_collector=self._metrics,
            logger=self._logger,
            refresh_interval=self._refresh,
        )

    def __repr__(self) -> str:
        return (
            f"AtwaterDashboard(backend={self._backend_name!r}, "
            f"refresh={self._refresh}s)"
        )
