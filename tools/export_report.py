#!/usr/bin/env python3
"""
tools/export_report.py — Generate a shareable markdown report from an Atwater run.

Loads JSONL logs + Optuna study + KB stats and produces a clean, executive-level
markdown report suitable for sharing with stakeholders or archiving in the repo.

Usage
-----
    python tools/export_report.py --study atwater --output REPORT.md
    python tools/export_report.py --logs logs/2026-04-22.jsonl --output REPORT.md
    python tools/export_report.py --study atwater --logs logs/*.jsonl --output REPORT.md
    python tools/export_report.py --study atwater          # prints to stdout
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root is on path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ---------------------------------------------------------------------------
# Log analysis helpers (duplicated lightly from analyze_run to keep standalone)
# ---------------------------------------------------------------------------


def _load_events(paths: list[Path]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    events.sort(key=lambda e: e.get("ts", ""))
    return events


def _filter(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [e for e in events if e.get("event_type") == event_type]


def _score_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    cycle_ends = _filter(events, "cycle_end")
    scores = [
        float(e["data"]["score"])
        for e in cycle_ends
        if e.get("data", {}).get("score") is not None
    ]
    if not scores:
        return {}
    best_idx = scores.index(max(scores))
    cycle_numbers = [
        e["data"].get("cycle") or e.get("cycle") or i + 1
        for i, e in enumerate(cycle_ends)
        if e.get("data", {}).get("score") is not None
    ]
    return {
        "count": len(scores),
        "first": scores[0],
        "last": scores[-1],
        "best": max(scores),
        "best_cycle": cycle_numbers[best_idx] if cycle_numbers else "?",
        "worst": min(scores),
        "avg": sum(scores) / len(scores),
        "improvement_abs": scores[-1] - scores[0],
        "improvement_pct": (
            (scores[-1] - scores[0]) / abs(scores[0]) * 100 if scores[0] != 0 else 0.0
        ),
        "scores": scores,
    }


def _kb_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import defaultdict

    writes = _filter(events, "knowledge_write")
    promotes = _filter(events, "knowledge_promote")
    by_tier: dict[str, int] = defaultdict(int)
    for e in writes:
        tier = e.get("data", {}).get("tier", "unknown")
        by_tier[tier] += 1

    # Extract sample entries (first N unique)
    samples: list[str] = []
    for e in writes[:10]:
        preview = e.get("data", {}).get("preview", "")
        if preview:
            samples.append(preview)

    return {
        "total_writes": len(writes),
        "total_promotions": len(promotes),
        "by_tier": dict(by_tier),
        "sample_entries": samples,
    }


def _token_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import defaultdict

    agent_evts = _filter(events, "agent_result")
    by_role: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0, "calls": 0})
    for e in agent_evts:
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
        "cost_est": total_in * 0.000003 + total_out * 0.000015,
    }


def _cascade_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    casc = _filter(events, "cascade_result")
    total = len(casc)
    sc = sum(1 for e in casc if e.get("data", {}).get("short_circuited", False))
    llm = sum(1 for e in casc if "llm" in e.get("data", {}).get("gates_passed", []))
    times = [float(e.get("data", {}).get("total_ms", 0)) for e in casc]
    return {
        "total": total,
        "short_circuit_rate": sc / total if total > 0 else 0.0,
        "llm_reach_rate": llm / total if total > 0 else 0.0,
        "avg_ms": sum(times) / len(times) if times else 0.0,
    }


def _error_summary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "cycle": e.get("cycle", "?"),
            "role": e.get("data", {}).get("role", "?"),
            "message": e.get("data", {}).get("message", "?"),
        }
        for e in _filter(events, "error")
    ]


def _param_history(events: list[dict[str, Any]]) -> dict[str, list[Any]]:
    from collections import defaultdict

    starts = _filter(events, "cycle_start")
    history: dict[str, list[Any]] = defaultdict(list)
    for e in starts:
        params = e.get("data", {}).get("params", {})
        for k, v in params.items():
            history[k].append(v)
    return dict(history)


# ---------------------------------------------------------------------------
# Optuna helpers
# ---------------------------------------------------------------------------


def _load_optuna_study(study_name: str, storage: str | None) -> Any:
    """Load an Optuna study. Returns None if Optuna is not available."""
    try:
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if storage:
            return optuna.load_study(study_name=study_name, storage=storage)
        # Try common locations
        for candidate in [
            f"sqlite:///optuna_trials.db",
            f"sqlite:///atwater_optuna.db",
        ]:
            try:
                return optuna.load_study(study_name=study_name, storage=candidate)
            except Exception:
                continue
        return None
    except ImportError:
        return None


def _optuna_summary(study: Any) -> dict[str, Any]:
    """Extract importances and best params from an Optuna study."""
    if study is None:
        return {}
    try:
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
    except Exception:
        best_params = {}
        best_score = None

    importances: dict[str, float] = {}
    try:
        from optuna.importance import get_param_importances

        importances = get_param_importances(study)
    except Exception:
        pass

    return {
        "total_trials": len(study.trials),
        "best_params": best_params,
        "best_score": best_score,
        "importances": importances,
    }


# ---------------------------------------------------------------------------
# Recommendations engine
# ---------------------------------------------------------------------------


def _generate_recommendations(
    score_data: dict[str, Any],
    kb_data: dict[str, Any],
    casc_data: dict[str, Any],
    tok_data: dict[str, Any],
    optuna_data: dict[str, Any],
    errors: list[dict[str, Any]],
) -> list[str]:
    recs: list[str] = []

    if score_data:
        improvement = score_data.get("improvement_abs", 0)
        if improvement < 0.05:
            recs.append(
                "Score improvement is modest (<5%). Consider expanding the Optuna "
                "search space or increasing diversity pressure."
            )
        elif improvement > 0.20:
            recs.append(
                "Strong score improvement observed. Run more cycles to see if "
                "the trajectory continues or plateaus."
            )

        scores = score_data.get("scores", [])
        if len(scores) > 10:
            recent_std = (
                sum((s - score_data["avg"]) ** 2 for s in scores[-10:]) / 10
            ) ** 0.5
            if recent_std < 0.01:
                recs.append(
                    "Scores appear to have converged (low recent variance). "
                    "Try forcing exploration or resetting the Optuna study."
                )

    sc_rate = casc_data.get("short_circuit_rate", 0)
    if sc_rate > 0.5:
        recs.append(
            f"Cascade short-circuit rate is high ({sc_rate:.0%}). Consider "
            "loosening fast/medium gate thresholds or reviewing output quality."
        )
    elif sc_rate < 0.1 and casc_data.get("total", 0) > 20:
        recs.append(
            "Cascade is reaching the LLM gate almost every time — consider "
            "tightening early gates to save cost."
        )

    kb_total = kb_data.get("total_writes", 0)
    if kb_total == 0:
        recs.append(
            "No knowledge base entries were written. Ensure the grader agent is "
            "configured to trigger knowledge writes on novel findings."
        )
    elif kb_total > 100:
        recs.append(
            f"Large KB ({kb_total} writes). Running consolidation more frequently "
            "may improve context relevance."
        )

    if errors:
        recs.append(
            f"{len(errors)} error event(s) detected. Review error details in the "
            "Errors section and check agent connectivity / LM Studio availability."
        )

    if optuna_data.get("importances"):
        top = list(optuna_data["importances"].items())[:1]
        if top:
            param, imp = top[0]
            recs.append(
                f"Parameter `{param}` has the highest Optuna importance ({imp:.2f}). "
                "Narrow its search range around the best known value for faster convergence."
            )

    if not recs:
        recs.append("Run looks healthy. Consider increasing cycles for further optimisation.")

    return recs


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def build_report(
    events: list[dict[str, Any]],
    log_paths: list[Path],
    study_name: str,
    study: Any,
    run_label: str = "",
) -> str:
    score_data = _score_summary(events)
    kb_data = _kb_summary(events)
    tok_data = _token_summary(events)
    casc_data = _cascade_summary(events)
    errors = _error_summary(events)
    optuna_data = _optuna_summary(study)
    param_hist = _param_history(events)
    recommendations = _generate_recommendations(
        score_data, kb_data, casc_data, tok_data, optuna_data, errors
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    session_ids = list({e.get("session_id", "?") for e in events})

    lines: list[str] = []

    # ── Title ──────────────────────────────────────────────────────────
    title = f"Atwater Run Report — {run_label or study_name}"
    lines += [f"# {title}\n", f"_Generated: {now}_  ", f"_Sessions: {', '.join(session_ids)}_\n"]

    # ── Executive Summary ──────────────────────────────────────────────
    lines.append("## Executive Summary\n")
    if score_data:
        n = score_data["count"]
        best = score_data["best"]
        first = score_data["first"]
        imp_abs = score_data["improvement_abs"]
        imp_pct = score_data["improvement_pct"]
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Cycles with scores | {n} |")
        lines.append(f"| First score | {first:.4f} |")
        lines.append(f"| Last score | {score_data['last']:.4f} |")
        lines.append(f"| **Best score** | **{best:.4f}** (cycle {score_data['best_cycle']}) |")
        lines.append(f"| Average score | {score_data['avg']:.4f} |")
        lines.append(f"| Improvement | {imp_abs:+.4f} ({imp_pct:+.1f}%) |")
        lines.append(f"| KB writes | {kb_data.get('total_writes', 0)} |")
        lines.append(f"| Cascade evaluations | {casc_data.get('total', 0)} |")
        lines.append(f"| Total tokens | {tok_data.get('total', 0):,} |")
        lines.append(f"| Errors | {len(errors)} |")
        lines.append("")
    else:
        lines.append("_No scored cycles found in the provided logs._\n")

    # ── Top Findings ───────────────────────────────────────────────────
    lines.append("## Top Findings\n")

    # Best params from Optuna or cycle history
    if optuna_data.get("best_params"):
        lines.append("### Best Parameter Combination (Optuna)\n")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for k, v in optuna_data["best_params"].items():
            lines.append(f"| `{k}` | `{v}` |")
        lines.append("")
    elif score_data and param_hist:
        lines.append("### Best Parameters (from log history)\n")
        lines.append("_Optuna study not loaded — showing most recent params._\n")
        best_cycle_idx = score_data["scores"].index(score_data["best"]) if score_data["scores"] else 0
        lines.append("| Parameter | Value at best cycle |")
        lines.append("|-----------|---------------------|")
        for k, vals in param_hist.items():
            v = vals[best_cycle_idx] if best_cycle_idx < len(vals) else (vals[-1] if vals else "N/A")
            lines.append(f"| `{k}` | `{v}` |")
        lines.append("")

    # Worst performing (lowest scores)
    if score_data and len(score_data["scores"]) > 1:
        lines.append("### Worst Scoring Cycles\n")
        scores_indexed = sorted(enumerate(score_data["scores"]), key=lambda x: x[1])[:3]
        lines.append("| Rank | Score |")
        lines.append("|------|-------|")
        for rank, (_, sc) in enumerate(scores_indexed, 1):
            lines.append(f"| {rank} | {sc:.4f} |")
        lines.append("")

    # ── Knowledge Base Highlights ──────────────────────────────────────
    lines.append("## Knowledge Base Highlights\n")
    if kb_data.get("total_writes", 0) > 0:
        lines.append(f"- **Total writes:** {kb_data['total_writes']}")
        lines.append(f"- **Total promotions:** {kb_data['total_promotions']}")
        if kb_data["by_tier"]:
            lines.append("- **Distribution:**")
            for tier, count in sorted(kb_data["by_tier"].items()):
                lines.append(f"  - `{tier}`: {count} entries")
        if kb_data["sample_entries"]:
            lines.append("\n**Sample entries (first 5 written):**\n")
            for i, entry in enumerate(kb_data["sample_entries"][:5], 1):
                lines.append(f"{i}. _{entry}_")
        lines.append("")
    else:
        lines.append("_No knowledge base entries written this run._\n")

    # ── Optuna Parameter Importances ───────────────────────────────────
    lines.append("## Optuna Parameter Importances\n")
    if optuna_data.get("importances"):
        lines.append("| Rank | Parameter | Importance | Bar |")
        lines.append("|------|-----------|------------|-----|")
        for rank, (param, imp) in enumerate(optuna_data["importances"].items(), 1):
            bar = "▪" * int(imp * 20)
            lines.append(f"| {rank} | `{param}` | {imp:.4f} | {bar} |")
        lines.append("")
        lines.append(
            f"_Based on {optuna_data.get('total_trials', '?')} Optuna trials._\n"
        )
    elif optuna_data.get("total_trials", 0) > 0:
        lines.append(
            f"_Not enough trials ({optuna_data['total_trials']}) for importance calculation._\n"
        )
    else:
        lines.append("_Optuna study not loaded or no trials available._\n")

    # ── Cascade & Token Stats ──────────────────────────────────────────
    lines.append("## Cascade Efficiency\n")
    if casc_data.get("total", 0) > 0:
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Evaluations | {casc_data['total']} |")
        lines.append(f"| Short-circuit rate | {casc_data['short_circuit_rate']:.0%} |")
        lines.append(f"| LLM gate reach rate | {casc_data['llm_reach_rate']:.0%} |")
        lines.append(f"| Avg cascade time | {casc_data['avg_ms']:.1f}ms |")
        lines.append("")
    else:
        lines.append("_No cascade events logged._\n")

    lines.append("## Token Usage\n")
    if tok_data.get("total", 0) > 0:
        lines.append("| Role | Calls | Tokens In | Tokens Out | Total |")
        lines.append("|------|-------|-----------|------------|-------|")
        for role, s in sorted(tok_data["by_role"].items()):
            lines.append(
                f"| {role} | {s['calls']} | {s['in']:,} | {s['out']:,} | {s['in']+s['out']:,} |"
            )
        lines.append("")
        lines.append(f"**Total:** {tok_data['total']:,} tokens  ")
        lines.append(f"**Estimated cost (OpenAI equiv.):** ${tok_data['cost_est']:.4f}\n")
    else:
        lines.append("_No agent token data logged._\n")

    # ── Errors ─────────────────────────────────────────────────────────
    if errors:
        lines.append("## Errors\n")
        lines.append("| Cycle | Role | Message |")
        lines.append("|-------|------|---------|")
        for e in errors[:20]:
            msg = str(e["message"])[:100].replace("|", "\\|")
            lines.append(f"| {e['cycle']} | {e['role']} | {msg} |")
        if len(errors) > 20:
            lines.append(f"\n_...and {len(errors) - 20} more errors_")
        lines.append("")

    # ── Recommendations ────────────────────────────────────────────────
    lines.append("## Recommendations for Next Run\n")
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"{i}. {rec}")
    lines.append("")

    # ── Footer ─────────────────────────────────────────────────────────
    lines.append("---")
    lines.append(f"_Atwater Cognitive Engine | Report generated by `tools/export_report.py`_")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="export_report",
        description="Generate a shareable markdown report from an Atwater run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--study",
        type=str,
        default="atwater-default",
        metavar="NAME",
        help="Optuna study name to load.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        metavar="URL",
        help="Optuna storage URL (e.g. sqlite:///optuna_trials.db). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--logs",
        nargs="*",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "JSONL log file(s) to include. "
            "Defaults to all .jsonl files in ./logs/ if not specified."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output file path (default: stdout).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        metavar="LABEL",
        help="Human-readable label for the run (used in report title).",
    )

    args = parser.parse_args(argv)

    # Resolve log files
    log_paths: list[Path] = []
    if args.logs:
        log_paths = [p for p in args.logs if p.exists()]
        missing = [p for p in args.logs if not p.exists()]
        for p in missing:
            print(f"[warn] Log file not found: {p}", file=sys.stderr)
    else:
        logs_dir = _project_root / "logs"
        if logs_dir.exists():
            log_paths = sorted(logs_dir.glob("*.jsonl"))
        if not log_paths:
            print("[warn] No log files found. Run with --logs to specify files.", file=sys.stderr)

    events = _load_events(log_paths) if log_paths else []

    # Load Optuna study
    study = _load_optuna_study(args.study, args.storage)
    if study is None:
        print(
            f"[info] Optuna study '{args.study}' not loaded "
            "(optuna not installed or study not found — continuing without it).",
            file=sys.stderr,
        )

    report = build_report(
        events=events,
        log_paths=log_paths,
        study_name=args.study,
        study=study,
        run_label=args.label,
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
