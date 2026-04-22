"""
main.py — Full cycle runner for the Atwater cognitive agent architecture.

Initialises all subsystems (SharedState, KnowledgeBase, Optuna study,
FlowController) and drives the evolution loop for N cycles, printing
progress after each cycle and a summary at the end.

Usage
-----
    python -m src.main --cycles 20 --study-name my-run --verbose
    python -m src.main --cycles 50 --config config/production.json
    python -m src.main --validate                  # health check, then exit
    python -m src.main --dry-run --cycles 5        # full cycle, no DB writes
    python -m src.main --resume --cycles 20        # resume from checkpoint
    python src/main.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import os as _os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure the project root (parent of src/) is on sys.path so that
# `from src.X import Y` works when running via `python3 src/main.py`.
_project_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---------------------------------------------------------------------------
# Logging setup (before any project imports so all loggers inherit it)
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )
    # Suppress noisy third-party loggers unless --verbose
    if not verbose:
        for noisy in ("optuna", "sentence_transformers", "transformers", "filelock"):
            logging.getLogger(noisy).setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atwater",
        description="Atwater cognitive agent architecture — production cycle runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=10,
        metavar="N",
        help="Number of production cycles to run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a JSON config file overriding defaults.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="atwater-default",
        metavar="NAME",
        help="Optuna study name (shared across runs for persistence).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging to stderr.",
    )
    parser.add_argument(
        "--state-db",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path for the SharedState SQLite database.",
    )
    parser.add_argument(
        "--knowledge-db",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path for the KnowledgeBase SQLite database.",
    )
    parser.add_argument(
        "--optuna-db",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path for the Optuna trials SQLite database.",
    )
    parser.add_argument(
        "--consolidation-interval",
        type=int,
        default=50,
        metavar="N",
        help="Run consolidation every N cycles.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        metavar="PATH",
        help="Directory to store cycle checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        metavar="N",
        help="Auto-checkpoint every N cycles.",
    )
    # -----------------------------------------------------------------------
    # Phase 7 flags
    # -----------------------------------------------------------------------
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Run full cycles but skip all database writes. "
            "Useful for testing without side effects."
        ),
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Run health checks, print a status report, and exit.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Load the most recent checkpoint and resume from that cycle number. "
            "Ignored if no checkpoint exists (starts from cycle 1)."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_config(config_path: Path | None) -> dict[str, Any]:
    """Load a JSON config file and return its contents as a dict."""
    if config_path is None:
        return {}
    if not config_path.exists():
        print(f"[warn] Config file not found: {config_path}", file=sys.stderr)
        return {}
    with config_path.open() as fh:
        return json.load(fh)


def _resolve_paths(
    args: argparse.Namespace,
    config: dict[str, Any],
    workspace: Path,
) -> tuple[Path, Path, Path]:
    """Resolve DB paths from args → config → defaults."""
    state_db = args.state_db or Path(config.get("state_db", workspace / "state.db"))
    knowledge_db = args.knowledge_db or Path(
        config.get("knowledge_db", workspace / "knowledge.db")
    )
    optuna_db = args.optuna_db or Path(
        config.get("optuna_db", workspace / "optuna_trials.db")
    )
    return state_db, knowledge_db, optuna_db


# ---------------------------------------------------------------------------
# Progress display helpers
# ---------------------------------------------------------------------------

def _progress_line(
    cycle: int,
    total: int,
    score: float | None,
    knowledge_writes: int,
    alerts: int,
    consolidated: bool,
    errors: dict[str, str],
    elapsed: float,
    dry_run: bool = False,
    checkpointed: bool = False,
) -> str:
    bar_width = 20
    filled = int(bar_width * cycle / total)
    bar = "█" * filled + "░" * (bar_width - filled)

    score_str = f"{score:.4f}" if score is not None else "  N/A "
    flags: list[str] = []
    if dry_run:
        flags.append("DRY-RUN")
    if consolidated:
        flags.append("CONSOLIDATED")
    if checkpointed:
        flags.append("CHECKPOINTED")
    if alerts:
        flags.append(f"{alerts} alert(s)")
    if errors:
        flags.append(f"ERRORS:{','.join(errors.keys())}")

    flag_str = "  [" + " | ".join(flags) + "]" if flags else ""
    return (
        f"  [{bar}] {cycle:>4}/{total}  "
        f"score={score_str}  kb_writes={knowledge_writes:>2}  "
        f"elapsed={elapsed:>6.1f}s{flag_str}"
    )


def _print_summary(
    results: list[Any],  # list[CycleResult]
    study: Any,           # optuna.Study
    kb: Any,              # KnowledgeBase
    total_elapsed: float,
    verbose: bool,
    dry_run: bool = False,
) -> None:
    """Print a structured post-run summary to stdout."""
    from src.optimization import get_importances, get_best_params, get_score_trend

    print("\n" + "=" * 60)
    print(f"  ATWATER — Run Summary{'  [DRY-RUN]' if dry_run else ''}")
    print("=" * 60)

    # Basic stats
    cycles_run = len(results)
    successful = sum(1 for r in results if r.success)
    failed = cycles_run - successful
    total_kb_writes = sum(len(r.knowledge_writes) for r in results)
    total_alerts = sum(len(r.diversity_alerts) for r in results)
    scores = [r.score for r in results if r.score is not None]

    print(f"\n  Cycles run   : {cycles_run}")
    print(f"  Successful   : {successful}")
    print(f"  Failed       : {failed}")
    print(f"  KB writes    : {total_kb_writes}")
    print(f"  Div. alerts  : {total_alerts}")
    print(f"  Wall time    : {total_elapsed:.1f}s")
    if dry_run:
        print(f"  Mode         : DRY-RUN (no DB writes)")

    # Score trend
    if scores:
        print(f"\n  Score trend  :")
        print(f"    First score : {scores[0]:.4f}")
        print(f"    Last score  : {scores[-1]:.4f}")
        print(f"    Best score  : {max(scores):.4f}")
        print(f"    Mean score  : {sum(scores) / len(scores):.4f}")

    # Optuna best params
    best_params = get_best_params(study)
    if best_params:
        print("\n  Best Optuna params:")
        for k, v in best_params.items():
            print(f"    {k:<20} = {v}")
    else:
        print("\n  Best Optuna params: (not yet available)")

    # Parameter importances
    importances = get_importances(study)
    if importances:
        print("\n  Top parameter importances:")
        for i, (param, imp) in enumerate(importances.items()):
            if i >= 5:
                print(f"    ... ({len(importances) - 5} more)")
                break
            bar = "▪" * int(imp * 20)
            print(f"    {param:<20} {imp:.4f}  {bar}")
    else:
        print("\n  Parameter importances: (not enough trials)")

    # Knowledge base size
    try:
        kb_repr = repr(kb)
        print(f"\n  Knowledge base: {kb_repr}")
    except Exception:
        print("\n  Knowledge base: (unavailable)")

    # Error summary
    error_cycles = [(r.cycle_number, r.errors) for r in results if r.errors]
    if error_cycles:
        print(f"\n  Errors ({len(error_cycles)} cycles had errors):")
        for cycle_num, errs in error_cycles[:5]:
            for role, msg in errs.items():
                print(f"    cycle={cycle_num} [{role}] {msg[:80]}")
        if len(error_cycles) > 5:
            print(f"    ... and {len(error_cycles) - 5} more error cycles")

    print("\n" + "=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Health check runner (--validate mode)
# ---------------------------------------------------------------------------

def _run_validate(
    state_db: Path,
    knowledge_db: Path,
    optuna_db: Path,
    lm_studio_url: str = "http://localhost:1234/v1",
) -> int:
    """
    Run all health checks, print a report, and return exit code.

    Returns 0 if all checks pass, 1 if any check fails.
    """
    from src.resilience.health_check import HealthChecker

    print("\nAtwater — Health Check\n")
    checker = HealthChecker(lm_studio_url=lm_studio_url)

    # Convert optuna_db Path to SQLite file path for disk check
    db_paths = [state_db, knowledge_db]
    # Optuna db may be "sqlite:///path/to/file.db" or a plain path
    optuna_path_str = str(optuna_db).replace("sqlite:///", "")
    db_paths.append(Path(optuna_path_str))

    results = checker.check_all(db_paths=db_paths)
    print(checker.report(results))
    print()

    all_healthy = all(r.healthy for r in results)
    return 0 if all_healthy else 1


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """
    Parse args, initialise all systems, run N cycles, print summary.

    Returns:
        Exit code (0 = success, 1 = initialisation failure).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    log = logging.getLogger("atwater.main")

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    config = _load_config(args.config)

    workspace = Path(__file__).parent.parent  # project root
    state_db, knowledge_db, optuna_db = _resolve_paths(args, config, workspace)

    study_name = config.get("study_name", args.study_name)
    cycles = config.get("cycles", args.cycles)
    consolidation_interval = config.get(
        "consolidation_interval", args.consolidation_interval
    )
    dry_run: bool = args.dry_run
    checkpoint_dir: Path = args.checkpoint_dir
    checkpoint_every: int = args.checkpoint_every

    # ------------------------------------------------------------------
    # --validate mode: run health checks and exit
    # ------------------------------------------------------------------
    if args.validate:
        return _run_validate(
            state_db=state_db,
            knowledge_db=knowledge_db,
            optuna_db=optuna_db,
        )

    # ------------------------------------------------------------------
    # Resilience subsystems
    # ------------------------------------------------------------------
    from src.resilience.checkpointing import CheckpointManager
    from src.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError
    from src.resilience.graceful_shutdown import ShutdownHandler

    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        save_every=checkpoint_every,
        keep_last=5,
    )

    circuit_breaker = CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=60,
        half_open_max=2,
    )

    shutdown_handler = ShutdownHandler(exit_on_shutdown=False)
    shutdown_handler.register()

    # ------------------------------------------------------------------
    # Print startup banner
    # ------------------------------------------------------------------
    print(f"\nAtwater — starting {cycles} cycle(s)  [study={study_name!r}]")
    print(f"  state_db     : {state_db}")
    print(f"  knowledge_db : {knowledge_db}")
    print(f"  optuna_db    : {optuna_db}")
    print(f"  consolidation: every {consolidation_interval} cycles")
    print(f"  checkpoints  : every {checkpoint_every} cycles → {checkpoint_dir}")
    if dry_run:
        print(f"  mode         : DRY-RUN (no DB writes)")
    if args.resume:
        print(f"  resume       : enabled")
    print()

    # ------------------------------------------------------------------
    # Initialise subsystems
    # ------------------------------------------------------------------
    log.debug("Initialising SharedState at %s", state_db)
    try:
        from src.memory import SharedState, KnowledgeBase
    except ImportError as exc:
        print(f"[error] Failed to import memory module: {exc}", file=sys.stderr)
        return 1

    try:
        shared_state = SharedState(db_path=state_db)
        log.debug("SharedState ready: %r", shared_state)
    except Exception as exc:
        print(f"[error] SharedState init failed: {exc}", file=sys.stderr)
        return 1

    log.debug("Initialising KnowledgeBase at %s", knowledge_db)
    try:
        knowledge_base = KnowledgeBase(db_path=knowledge_db)
        log.debug("KnowledgeBase ready: %r", knowledge_base)
    except Exception as exc:
        print(f"[error] KnowledgeBase init failed: {exc}", file=sys.stderr)
        return 1

    log.debug("Initialising Optuna study %r at %s", study_name, optuna_db)
    try:
        from src.optimization import create_study
        study = create_study(name=study_name, storage_path=str(optuna_db))
        log.debug("Optuna study ready (trials so far: %d)", len(study.trials))
    except Exception as exc:
        print(f"[error] Optuna study init failed: {exc}", file=sys.stderr)
        return 1

    log.debug("Initialising FlowController")
    try:
        from src.orchestrator.flow_controller import FlowController
        flow = FlowController(
            shared_state=shared_state,
            knowledge_base=knowledge_base,
            study=study,
            consolidation_interval=consolidation_interval,
        )
        log.debug("FlowController ready")
    except Exception as exc:
        print(f"[error] FlowController init failed: {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Register shutdown callback: save checkpoint on SIGTERM/SIGINT
    # ------------------------------------------------------------------
    _last_result_ref: list[Any] = []  # mutable container for closure

    def _on_shutdown() -> None:
        log.warning("Shutdown requested — saving emergency checkpoint")
        if _last_result_ref:
            last = _last_result_ref[-1]
            checkpoint_mgr.save_checkpoint(
                cycle_number=last.cycle_number,
                state={"last_score": last.score, "params": last.params_used},
                study_name=study_name,
                kb_stats={},
                force=True,
            )
        try:
            shared_state.close()
            knowledge_base.close()
        except Exception:
            pass

    shutdown_handler.on_shutdown(_on_shutdown)

    # ------------------------------------------------------------------
    # --resume: load checkpoint and determine start cycle
    # ------------------------------------------------------------------
    start_cycle = 1
    if args.resume:
        checkpoint = checkpoint_mgr.load_checkpoint()
        if checkpoint:
            start_cycle = checkpoint.cycle_number + 1
            print(
                f"  Resuming from cycle {start_cycle} "
                f"(checkpoint at cycle {checkpoint.cycle_number})\n"
            )
        else:
            print("  No checkpoint found — starting from cycle 1\n")

    # ------------------------------------------------------------------
    # Run cycles
    # ------------------------------------------------------------------
    run_start = time.perf_counter()
    results: list[Any] = []

    # Compute effective total for the progress bar
    remaining_cycles = cycles - (start_cycle - 1)
    print(f"  Running {remaining_cycles} cycle(s) (start={start_cycle})...\n")

    for i in range(start_cycle, start_cycle + remaining_cycles):
        # --- Check for shutdown signal before each cycle ---
        if shutdown_handler.is_shutdown_requested():
            print(f"\n  Shutdown requested at cycle {i} — stopping.", file=sys.stderr)
            break

        cycle_start = time.perf_counter()
        try:
            # Wrap the cycle in the circuit breaker to protect against
            # cascading LLM failures.
            def _run_cycle_inner(cycle_number: int) -> Any:
                return flow.run_cycle(cycle_number=cycle_number)

            try:
                result = circuit_breaker.call(_run_cycle_inner, i)
            except CircuitOpenError as coe:
                log.error("Circuit breaker OPEN at cycle %d: %s", i, coe)
                from src.orchestrator.flow_controller import CycleResult
                result = CycleResult(
                    cycle_number=i,
                    params_used={},
                    score=None,
                    knowledge_writes=[],
                    diversity_alerts=[],
                    consolidated=False,
                    success=False,
                    errors={"circuit_breaker": str(coe)},
                )

        except Exception as exc:
            log.error("Unhandled exception in cycle %d: %s", i, exc, exc_info=True)
            from src.orchestrator.flow_controller import CycleResult
            result = CycleResult(
                cycle_number=i,
                params_used={},
                score=None,
                knowledge_writes=[],
                diversity_alerts=[],
                consolidated=False,
                success=False,
                errors={"unhandled": str(exc)},
            )

        results.append(result)
        _last_result_ref.append(result)  # keep most recent for shutdown callback
        if len(_last_result_ref) > 1:
            _last_result_ref.pop(0)

        cycle_elapsed = time.perf_counter() - cycle_start
        total_elapsed = time.perf_counter() - run_start

        # --- Auto-checkpoint ---
        saved_path = None
        if not dry_run:
            kb_stats: dict[str, Any] = {}
            try:
                kb_stats = {"repr": repr(knowledge_base)}
            except Exception:
                pass
            saved_path = checkpoint_mgr.save_checkpoint(
                cycle_number=i,
                state={
                    "last_score": result.score,
                    "params": result.params_used,
                    "errors": result.errors,
                },
                study_name=study_name,
                kb_stats=kb_stats,
            )

        line = _progress_line(
            cycle=i - start_cycle + 1,
            total=remaining_cycles,
            score=result.score,
            knowledge_writes=len(result.knowledge_writes),
            alerts=len(result.diversity_alerts),
            consolidated=result.consolidated,
            errors=result.errors,
            elapsed=total_elapsed,
            dry_run=dry_run,
            checkpointed=saved_path is not None,
        )
        print(line)

    total_elapsed = time.perf_counter() - run_start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _print_summary(
        results=results,
        study=study,
        kb=knowledge_base,
        total_elapsed=total_elapsed,
        verbose=args.verbose,
        dry_run=dry_run,
    )

    # ------------------------------------------------------------------
    # Final checkpoint on clean exit
    # ------------------------------------------------------------------
    if results and not dry_run:
        checkpoint_mgr.save_checkpoint(
            cycle_number=results[-1].cycle_number,
            state={
                "last_score": results[-1].score,
                "params": results[-1].params_used,
            },
            study_name=study_name,
            kb_stats={},
            force=True,
        )

    # ------------------------------------------------------------------
    # Clean shutdown
    # ------------------------------------------------------------------
    log.debug("Closing SharedState connection")
    try:
        shared_state.close()
    except Exception as exc:
        log.warning("SharedState close error: %s", exc)

    log.debug("Closing KnowledgeBase connection")
    try:
        knowledge_base.close()
    except Exception as exc:
        log.warning("KnowledgeBase close error: %s", exc)

    log.debug("Shutdown complete")
    cb_stats = circuit_breaker.get_stats()
    if cb_stats["total_trips"] > 0:
        print(
            f"  [circuit-breaker] tripped {cb_stats['total_trips']} time(s) "
            f"({cb_stats['total_failures']} failures across {cb_stats['total_calls']} calls)",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
