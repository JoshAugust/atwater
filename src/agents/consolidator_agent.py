"""
src.agents.consolidator_agent — Consolidation Agent.

Responsibility
--------------
The bridge between statistical truth (Optuna) and qualitative knowledge (KB).

Runs periodically (every N cycles, configurable) to:
1. Pull parameter importances from Optuna.
2. Interpret them into natural-language knowledge entries.
3. Hand the full KB entry list + Optuna stats to ConsolidationEngine.
4. Write the resulting promotions, merges, and archives back to the KB.

This agent does NOT call an LLM directly.  It uses the ConsolidationEngine
for rule-based consolidation and returns an interpretation prompt in
AgentResult.output for the orchestrator to run through an LLM when
qualitative synthesis is needed.

State contract
--------------
Reads:  (none — reads directly from KB and Optuna, orchestrator passes context)
Writes: (none — all writes go through knowledge_writes in AgentResult)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import optuna

from src.optimization import get_importances, get_best_params, get_dimension_stats
from src.agents.base import AgentBase, AgentContext, AgentResult

from src.knowledge.consolidator import ConsolidationEngine
from src.knowledge.models import KnowledgeEntry, KnowledgeTier

logger = logging.getLogger(__name__)

DEFAULT_CONSOLIDATION_INTERVAL: int = 50  # run every N cycles


class ConsolidatorAgent(AgentBase):
    """
    Consolidation Agent — periodic knowledge compaction and Optuna interpretation.

    Parameters
    ----------
    study : optuna.Study
        Active Optuna study for reading importances and trial stats.
    consolidation_engine : ConsolidationEngine | None
        Instance of the KB consolidation engine.  Created with defaults if None.
    interval : int
        Run consolidation every N cycles.  Default 50.
    importance_confidence_scale : float
        Multiply Optuna importance values by this factor to derive KB confidence.
        Clamped to [0.0, 1.0].  Default 1.0.
    """

    def __init__(
        self,
        study: optuna.Study,
        consolidation_engine: ConsolidationEngine | None = None,
        interval: int = DEFAULT_CONSOLIDATION_INTERVAL,
        importance_confidence_scale: float = 1.0,
    ) -> None:
        self._study = study
        self._engine = consolidation_engine or ConsolidationEngine()
        self._interval = interval
        self._importance_confidence_scale = importance_confidence_scale

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "ConsolidatorAgent"

    @property
    def role(self) -> str:
        return "consolidator"

    @property
    def readable_state_keys(self) -> list[str]:
        return []  # Reads directly from Optuna + KB via context

    @property
    def writable_state_keys(self) -> list[str]:
        return []  # All writes go through knowledge_writes

    # ------------------------------------------------------------------
    # Core execute
    # ------------------------------------------------------------------

    def execute(self, context: AgentContext) -> AgentResult:
        """
        Run a consolidation pass.

        Steps
        -----
        1. Check if this cycle is due for consolidation.
        2. Pull Optuna importances + dimension stats.
        3. Convert importances → draft KB observation entries.
        4. Run ConsolidationEngine over the existing KB entries (from context).
        5. Collect knowledge_writes: new importance entries + changelog entries.
        6. Return a synthesis prompt for the orchestrator to optionally run.

        The orchestrator decides whether to run the synthesis prompt (cost vs
        benefit trade-off) — the agent does not assume it will be called.
        """
        current_cycle: int = context.scoped_state.get("current_cycle", 0)
        existing_entries: list[dict[str, Any]] = context.knowledge_entries

        if not self._is_due(current_cycle):
            logger.debug(
                "[ConsolidatorAgent] Skipping — cycle %d, interval %d.",
                current_cycle,
                self._interval,
            )
            return AgentResult(
                output={"skipped": True, "reason": "not_due", "current_cycle": current_cycle},
                state_updates={},
                knowledge_writes=[],
                score=None,
            )

        logger.info(
            "[ConsolidatorAgent] Running consolidation pass at cycle %d.", current_cycle
        )

        # --- Step 2: Pull Optuna statistics ---
        importances = self._safe_get_importances()
        best_params = self._safe_get_best_params()
        dimension_stats = self._safe_get_dimension_stats()

        # --- Step 3: Convert importances to draft KB entries ---
        importance_entries = self._importances_to_knowledge(importances, best_params, current_cycle)

        # --- Step 4: Run ConsolidationEngine ---
        changelog = self._run_consolidation_engine(existing_entries, current_cycle)

        # --- Step 5: Collect knowledge_writes ---
        knowledge_writes: list[dict[str, Any]] = []
        knowledge_writes.extend(importance_entries)

        # Convert changelog entries to KB writes (promoting/merging summaries)
        for change in changelog.changes if hasattr(changelog, "changes") else []:
            kw = self._changelog_entry_to_kb_write(change, current_cycle)
            if kw:
                knowledge_writes.append(kw)

        # --- Step 6: Build synthesis prompt ---
        synthesis_prompt = self._build_synthesis_prompt(
            importances, best_params, dimension_stats, changelog
        )

        output = {
            "skipped": False,
            "current_cycle": current_cycle,
            "importance_entries_written": len(importance_entries),
            "changelog": str(changelog),
            "best_params": best_params,
            "importances": importances,
            "synthesis_prompt": synthesis_prompt,
        }

        logger.info(
            "[ConsolidatorAgent] Consolidation complete. Wrote %d KB entries, changelog: %s",
            len(knowledge_writes),
            str(changelog)[:200],
        )

        return AgentResult(
            output=output,
            state_updates={},
            knowledge_writes=knowledge_writes,
            score=None,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _is_due(self, current_cycle: int) -> bool:
        """Return True if this cycle is a consolidation milestone."""
        if current_cycle == 0:
            return False
        return current_cycle % self._interval == 0

    def _safe_get_importances(self) -> dict[str, float]:
        try:
            return get_importances(self._study)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ConsolidatorAgent] Failed to get importances: %s", exc)
            return {}

    def _safe_get_best_params(self) -> dict[str, Any]:
        try:
            return get_best_params(self._study)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ConsolidatorAgent] Failed to get best params: %s", exc)
            return {}

    def _safe_get_dimension_stats(self) -> dict[str, Any]:
        try:
            return get_dimension_stats(self._study)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ConsolidatorAgent] Failed to get dimension stats: %s", exc)
            return {}

    def _importances_to_knowledge(
        self,
        importances: dict[str, float],
        best_params: dict[str, Any],
        current_cycle: int,
    ) -> list[dict[str, Any]]:
        """
        Convert Optuna parameter importances into KB observation entries.

        High-importance parameters get higher confidence.  Best param values
        are embedded as metadata for downstream use.
        """
        entries: list[dict[str, Any]] = []
        if not importances:
            return entries

        # Sort by importance descending
        ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)

        for rank, (param, importance) in enumerate(ranked, 1):
            confidence = min(1.0, max(0.0, importance * self._importance_confidence_scale))
            best_value = best_params.get(param, "unknown")

            content = (
                f"Parameter '{param}' has statistical importance {importance:.3f} "
                f"(rank #{rank} of {len(ranked)}) as of cycle {current_cycle}. "
                f"Best observed value: {best_value}."
            )

            kw = {
                "content": content,
                "tier": "observation",
                "confidence": round(confidence, 4),
                "topic_cluster": f"optuna_importance_{param}",
                "metadata": {
                    "param": param,
                    "importance": importance,
                    "rank": rank,
                    "best_value": best_value,
                    "cycle": current_cycle,
                    "source": "consolidator_optuna_interpretation",
                },
            }
            self.validate_knowledge_write(kw)
            entries.append(kw)

        return entries

    def _run_consolidation_engine(
        self,
        existing_entries: list[dict[str, Any]],
        current_cycle: int,
    ) -> Any:
        """
        Convert context dict entries back to KnowledgeEntry objects and run
        the ConsolidationEngine.  Returns the changelog.
        """
        kb_entries: list[KnowledgeEntry] = []
        for raw in existing_entries:
            try:
                entry = self._dict_to_knowledge_entry(raw)
                kb_entries.append(entry)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[ConsolidatorAgent] Could not reconstruct KnowledgeEntry: %s — %s",
                    raw.get("id", "?"),
                    exc,
                )

        if not kb_entries:
            logger.info("[ConsolidatorAgent] No KB entries to consolidate.")
            return None

        # Run without embeddings (ConsolidationEngine handles missing embeddings gracefully)
        try:
            changelog = self._engine.run_consolidation(kb_entries, current_cycle=current_cycle)
            return changelog
        except Exception as exc:  # noqa: BLE001
            logger.error("[ConsolidatorAgent] ConsolidationEngine error: %s", exc)
            return None

    def _dict_to_knowledge_entry(self, raw: dict[str, Any]) -> KnowledgeEntry:
        """Reconstruct a KnowledgeEntry from a dict (as stored in context)."""
        return KnowledgeEntry(
            id=raw.get("id", ""),
            content=raw.get("content", ""),
            tier=raw.get("tier", "observation"),
            confidence=float(raw.get("confidence", 0.5)),
            created_cycle=int(raw.get("created_cycle", 0)),
            last_validated_cycle=int(raw.get("last_validated_cycle", 0)),
            validation_count=int(raw.get("validation_count", 0)),
            contradicted_by=list(raw.get("contradicted_by", [])),
            optuna_evidence=dict(raw.get("optuna_evidence", {})),
            topic_cluster=str(raw.get("topic_cluster", "general")),
        )

    def _changelog_entry_to_kb_write(
        self, change: Any, current_cycle: int
    ) -> dict[str, Any] | None:
        """Convert a changelog entry into a KB write dict if it's a promotion."""
        try:
            action = getattr(change, "action", None) or change.get("action", "")
            if action not in ("promoted", "merged"):
                return None

            content = getattr(change, "summary", None) or change.get("summary", str(change))
            tier = getattr(change, "new_tier", None) or change.get("new_tier", "pattern")

            return {
                "content": f"[Consolidation cycle {current_cycle}] {content}",
                "tier": tier,
                "confidence": 0.75,
                "topic_cluster": "consolidation_result",
                "metadata": {
                    "action": action,
                    "cycle": current_cycle,
                    "source": "consolidation_engine",
                },
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("[ConsolidatorAgent] Could not convert changelog entry: %s", exc)
            return None

    def _build_synthesis_prompt(
        self,
        importances: dict[str, float],
        best_params: dict[str, Any],
        dimension_stats: dict[str, Any],
        changelog: Any,
    ) -> str:
        """
        Build an optional LLM synthesis prompt that the orchestrator can run
        to produce qualitative interpretations of the Optuna statistics.
        """
        importance_lines = "\n".join(
            f"  {param}: {score:.3f}" for param, score in sorted(importances.items(), key=lambda x: -x[1])
        ) or "  (none available)"

        best_lines = "\n".join(
            f"  {k}: {v}" for k, v in best_params.items()
        ) or "  (none available)"

        changelog_str = str(changelog)[:500] if changelog else "(no consolidation performed)"

        return (
            "You are a knowledge synthesis expert. Below are statistical findings "
            "from an Optuna optimization study. Interpret these findings into "
            "actionable qualitative insights for the knowledge base.\n\n"
            "Parameter Importances (higher = more impact on output quality):\n"
            f"{importance_lines}\n\n"
            "Best Observed Parameters:\n"
            f"{best_lines}\n\n"
            "Consolidation Changelog:\n"
            f"{changelog_str}\n\n"
            "Instructions:\n"
            "1. For each high-importance parameter (importance > 0.3), write a "
            "one-sentence insight explaining what this means qualitatively.\n"
            "2. Identify any surprising findings (low importance for expected params, etc.).\n"
            "3. Suggest any patterns or rules that could be elevated to higher KB tiers.\n"
            "Output as a structured list of insights."
        )
