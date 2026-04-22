#!/usr/bin/env python3
"""
tools/analyze_run.py — Post-hoc run analysis for Atwater JSONL logs.

Loads one or more JSONL log files produced by AtwaterLogger and generates
a comprehensive markdown report including:

  - Score improvement curve (ASCII chart or matplotlib if available)
  - Knowledge base growth / compaction over time
  - Parameter convergence (when did params stabilise?)
  - Cascade efficiency over time
  - Token usage breakdown by agent

Usage
-----
    python tools/analyze_run.py logs/2026-04-22.jsonl
    python tools/analyze_run.py logs/*.jsonl --output report.md
    python tools/analyze_run.py logs/2026-04-22.jsonl --no-charts
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Generator

# Ensure project root on path when run directly
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------


def load_events(paths: list[Path]) -> list[dict[str, Any]]:
    """Load and merge events from one or more JSONL files, sorted by timestamp."""
    events: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    # Sort by timestamp string (ISO-8601 sorts lexicographically)
    events.sort(key=lambda e: e.get("ts", ""))
    return events


def filter_events(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [e for e in events if e.get("event_type") == event_type]


# ---------------------------------------------------------------------------
# ASCII charts
# ---------------------------------------------------------------------------

_PLOT_CHARS = " ▁▂▃▄▅▆▇█"


def ascii_chart(
    values: list[float],
    width: int = 60,
    height: int = 8,
    title: str = "",
    y_label: str = "score",
) -> str:
    """Render a compact ASCII line chart. Returns a multi-line string."""
    if not values:
        return "(no data)\n"

    # Downsample to width
    n = len(values)
    if n > width:
        step = n / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values

    lo, hi = min(sampled), max(sampled)
    span = hi - lo if hi > lo else 1.0

    rows: list[str] = []
    if title:
        rows.append(f"  {title}")
    rows.append(f"  {hi:.3f} ┐")
    for row in range(height - 1, -1, -1):
        threshold = lo + (row / (height - 1)) * span
        line = ""
        for v in sampled:
            if v >= threshold:
                line += "█"
            else:
                line += " "
        y_tick = f"{lo + (row / (height - 1)) * span:.3f}" if row in (0, height - 1) else "      "
        rows.append(f"  {y_tick} │{line}│")
    rows.append(f"  {lo:.3f} ┘")
    rows.append(f"  {'└' + '─' * len(sampled) + '┘'}")
    rows.append(f"  {0:<6}  cycle  {n - 1}")
    return "\n".join(rows) + "\n"


def _try_matplotlib(
    x: list[Any],
    y: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path | None = None,
) -> str | None:
    """Try to generate a matplotlib figure. Returns path string or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, y, linewidth=1.5, color="#2ea44f")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        save_path = out_path or Path(f"/tmp/atwater_{title.replace(' ', '_').lower()}.png")
        fig.savefig(save_path, dpi=100)
        plt.close(fig)
        return str(save_path)
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------


def analyze_scores(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract per-cycle score data."""
    cycle_ends = filter_events(events, "cycle_end")
    cycles, scores = [], []
    for e in cycle_ends:
        d = e.get("data", {})
        cycle = d.get("cycle") or e.get("cycle") or 0
        score = d.get("score")
        if score is not None:
            cycles.append(cycle)
            scores.append(float(score))

    if not scores:
        return {"cycles": [], "scores": [], "best": None, "worst": None, "trend": []}

    best_idx = scores.index(max(scores))
    worst_idx = scores.index(min(scores))

    # Rolling best (improvement curve)
    rolling_best = []
    current_best = -float("inf")
    for s in scores:
        if s > current_best:
            current_best = s
        rolling_best.append(current_best)

    return {
        "cycles": cycles,
        "scores": scores,
        "rolling_best": rolling_best,
        "best": {"cycle": cycles[best_idx], "score": scores[best_idx]},
        "worst": {"cycle": cycles[worst_idx], "score": scores[worst_idx]},
        "first": scores[0],
        "last": scores[-1],
        "improvement": scores[-1] - scores[0] if len(scores) > 1 else 0.0,
        "avg": sum(scores) / len(scores),
    }


def analyze_knowledge(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Track KB writes and promotions over time."""
    writes = filter_events(events, "knowledge_write")
    promotes = filter_events(events, "knowledge_promote")

    # Cumulative writes per tier over cycles
    tier_timeline: dict[str, list[int]] = defaultdict(list)
    tier_counts: dict[str, int] = defaultdict(int)

    for e in writes:
        d = e.get("data", {})
        tier = d.get("tier", "unknown")
        tier_counts[tier] += 1

    for e in promotes:
        d = e.get("data", {})
        to_tier = d.get("to_tier", "unknown")
        tier_counts[to_tier] += 1

    return {
        "total_writes": len(writes),
        "total_promotions": len(promotes),
        "by_tier": dict(tier_counts),
        "write_cycles": [e.get("cycle") for e in writes],
    }


def analyze_params(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyse parameter convergence over time."""
    cycle_ends = filter_events(events, "cycle_end")
    optuna_trials = filter_events(events, "optuna_trial")

    # Collect param values per cycle from cycle_start or optuna_trial
    starts = filter_events(events, "cycle_start")
    param_history: dict[str, list[tuple[int, Any]]] = defaultdict(list)

    for e in starts:
        d = e.get("data", {})
        cycle = d.get("cycle") or e.get("cycle") or 0
        params = d.get("params", {})
        for k, v in params.items():
            param_history[k].append((cycle, v))

    for e in optuna_trials:
        d = e.get("data", {})
        cycle = e.get("cycle") or 0
        params = d.get("params", {})
        for k, v in params.items():
            param_history[k].append((cycle, v))

    # Detect stabilisation: last cycle where param changed by > 5%
    convergence: dict[str, int | None] = {}
    for param, history in param_history.items():
        if len(history) < 2:
            convergence[param] = None
            continue
        last_change = None
        prev = history[0][1]
        for cycle, val in history[1:]:
            try:
                if abs(float(val) - float(prev)) > abs(float(prev)) * 0.05 + 1e-6:
                    last_change = cycle
                prev = val
            except (TypeError, ValueError):
                if val != prev:
                    last_change = cycle
                prev = val
        convergence[param] = last_change

    return {
        "param_history": {k: v for k, v in param_history.items()},
        "convergence_cycle": convergence,
        "unique_params": list(param_history.keys()),
    }


def analyze_cascade(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Cascade efficiency statistics over time."""
    cascade_events = filter_events(events, "cascade_result")

    total = len(cascade_events)
    short_circuited = sum(
        1 for e in cascade_events if e.get("data", {}).get("short_circuited", False)
    )
    fast_pass = sum(
        1 for e in cascade_events if "fast" in e.get("data", {}).get("gates_passed", [])
    )
    medium_pass = sum(
        1 for e in cascade_events if "medium" in e.get("data", {}).get("gates_passed", [])
    )
    llm_reach = sum(
        1 for e in cascade_events if "llm" in e.get("data", {}).get("gates_passed", [])
    )

    times_ms = [
        float(e.get("data", {}).get("total_ms", 0)) for e in cascade_events
    ]
    avg_ms = sum(times_ms) / len(times_ms) if times_ms else 0.0

    return {
        "total": total,
        "short_circuited": short_circuited,
        "short_circuit_rate": short_circuited / total if total > 0 else 0.0,
        "fast_pass": fast_pass,
        "medium_pass": medium_pass,
        "llm_reach": llm_reach,
        "avg_ms": avg_ms,
        # Rolling short-circuit rate (window 10)
        "sc_over_time": [
            sum(
                1
                for e in cascade_events[max(0, i - 10) : i + 1]
                if e.get("data", {}).get("short_circuited", False)
            )
            / min(i + 1, 10)
            for i in range(total)
        ],
    }


def analyze_tokens(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Token usage breakdown by agent."""
    agent_events = filter_events(events, "agent_result")

    by_role: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0, "calls": 0})
    for e in agent_events:
        d = e.get("data", {})
        role = d.get("role", "unknown")
        by_role[role]["in"] += d.get("tokens_in", 0)
        by_role[role]["out"] += d.get("tokens_out", 0)
        by_role[role]["calls"] += 1

    total_in = sum(v["in"] for v in by_role.values())
    total_out = sum(v["out"] for v in by_role.values())

    return {
        "by_role": {k: dict(v) for k, v in by_role.items()},
        "total_in": total_in,
        "total_out": total_out,
        "total": total_in + total_out,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    paths: list[Path],
    events: list[dict[str, Any]],
    use_charts: bool = True,
    output: Path | None = None,
) -> str:
    """Generate a full markdown report string."""
    lines: list[str] = []

    # ---- Header ----
    session_ids = list({e.get("session_id", "?") for e in events})
    session_str = ", ".join(session_ids) if session_ids else "unknown"
    log_files = ", ".join(str(p) for p in paths)

    lines.append("# Atwater Run Analysis Report\n")
    lines.append(f"**Log files:** `{log_files}`  ")
    lines.append(f"**Sessions:** {session_str}  ")
    lines.append(f"**Total events:** {len(events)}  \n")

    # ---- Score analysis ----
    score_data = analyze_scores(events)
    lines.append("## Score Improvement\n")
    if score_data["scores"]:
        lines.append(f"- **First score:** {score_data['first']:.4f}")
        lines.append(f"- **Last score:** {score_data['last']:.4f}")
        lines.append(f"- **Best:** {score_data['best']['score']:.4f} (cycle {score_data['best']['cycle']})")
        lines.append(f"- **Worst:** {score_data['worst']['score']:.4f} (cycle {score_data['worst']['cycle']})")
        lines.append(f"- **Average:** {score_data['avg']:.4f}")
        lines.append(
            f"- **Improvement:** "
            f"{score_data['improvement']:+.4f} "
            f"({(score_data['improvement'] / score_data['first'] * 100):+.1f}% from first)"
            if score_data["first"] != 0
            else f"- **Improvement:** {score_data['improvement']:+.4f}"
        )
        lines.append("")

        if use_charts:
            chart = ascii_chart(
                score_data["rolling_best"],
                width=60,
                height=8,
                title="Rolling best score",
                y_label="score",
            )
            lines.append("```")
            lines.append(chart)
            lines.append("```\n")
    else:
        lines.append("_No cycle_end events with scores found._\n")

    # ---- Knowledge growth ----
    kb_data = analyze_knowledge(events)
    lines.append("## Knowledge Base Growth\n")
    lines.append(f"- **Total writes:** {kb_data['total_writes']}")
    lines.append(f"- **Total promotions:** {kb_data['total_promotions']}")
    if kb_data["by_tier"]:
        lines.append("- **By tier:**")
        for tier, count in sorted(kb_data["by_tier"].items()):
            lines.append(f"  - `{tier}`: {count}")
    lines.append("")

    # ---- Parameter convergence ----
    param_data = analyze_params(events)
    lines.append("## Parameter Convergence\n")
    if param_data["unique_params"]:
        lines.append("| Parameter | Last Change (cycle) |")
        lines.append("|-----------|---------------------|")
        for param in sorted(param_data["unique_params"]):
            last = param_data["convergence_cycle"].get(param)
            last_str = str(last) if last is not None else "stable from start"
            lines.append(f"| `{param}` | {last_str} |")
        lines.append("")
    else:
        lines.append("_No parameter history found (no cycle_start or optuna_trial events)._\n")

    # ---- Cascade efficiency ----
    casc_data = analyze_cascade(events)
    lines.append("## Cascade Efficiency\n")
    if casc_data["total"] > 0:
        lines.append(f"- **Total evaluated:** {casc_data['total']}")
        lines.append(f"- **Short-circuited:** {casc_data['short_circuited']} ({casc_data['short_circuit_rate']:.0%})")
        lines.append(f"- **Fast gate pass:** {casc_data['fast_pass']}")
        lines.append(f"- **Medium gate pass:** {casc_data['medium_pass']}")
        lines.append(f"- **LLM gate reached:** {casc_data['llm_reach']}")
        lines.append(f"- **Avg cascade time:** {casc_data['avg_ms']:.1f}ms")
        lines.append("")
        if use_charts and len(casc_data["sc_over_time"]) > 1:
            sc_vals = [v * 100 for v in casc_data["sc_over_time"]]
            chart = ascii_chart(
                sc_vals, width=60, height=6, title="Short-circuit rate % (rolling window 10)", y_label="%"
            )
            lines.append("```")
            lines.append(chart)
            lines.append("```\n")
    else:
        lines.append("_No cascade_result events found._\n")

    # ---- Token usage ----
    tok_data = analyze_tokens(events)
    lines.append("## Token Usage by Agent\n")
    if tok_data["by_role"]:
        lines.append("| Role | Calls | Tokens In | Tokens Out | Total |")
        lines.append("|------|-------|-----------|------------|-------|")
        for role, stats in sorted(tok_data["by_role"].items()):
            total = stats["in"] + stats["out"]
            lines.append(
                f"| {role} | {stats['calls']} | {stats['in']:,} | {stats['out']:,} | {total:,} |"
            )
        lines.append("")
        lines.append(f"**Grand total:** {tok_data['total']:,} tokens "
                     f"(in={tok_data['total_in']:,}, out={tok_data['total_out']:,})")
        # Rough cost
        est = tok_data['total_in'] * 0.000003 + tok_data['total_out'] * 0.000015
        lines.append(f"**Estimated cost (OpenAI equiv.):** ${est:.4f}")
        lines.append("")
    else:
        lines.append("_No agent_result events found._\n")

    # ---- Footer ----
    from datetime import datetime, timezone
    lines.append("---")
    lines.append(f"_Generated by `tools/analyze_run.py` at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="analyze_run",
        description="Post-hoc analysis of Atwater JSONL run logs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "logs",
        nargs="+",
        type=Path,
        metavar="LOG",
        help="Path(s) to JSONL log file(s).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write markdown report to this file (default: stdout).",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        default=False,
        help="Suppress ASCII chart rendering.",
    )

    args = parser.parse_args(argv)

    # Validate inputs
    missing = [p for p in args.logs if not p.exists()]
    if missing:
        for p in missing:
            print(f"[error] File not found: {p}", file=sys.stderr)
        return 1

    events = load_events(args.logs)
    if not events:
        print("[warn] No events found in the provided log files.", file=sys.stderr)

    report = generate_report(
        paths=args.logs,
        events=events,
        use_charts=not args.no_charts,
        output=args.output,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Report written to: {args.output}")
    else:
        print(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
