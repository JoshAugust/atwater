"""
src.llm.prompts — Prompt templates for each Atwater agent role.

Design principles:
- System prompts are static strings (~200 tokens each). They define the agent's
  identity, authority, and core decision rules.
- Builder functions assemble the full message list dynamically by injecting
  scoped state, knowledge entries, and Optuna context.
- All prompts produce OpenAI-format message lists: list[dict[str, str]].
- Keep prompts concrete and action-oriented, not vague. Every agent knows
  exactly what to produce and in what format.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# System prompts — one per agent role
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {
    "director": (
        "You are the Director Engine for the Atwater cognitive architecture. "
        "Your sole purpose is to select the next parameter combination to test, "
        "grounded in statistical evidence from Optuna and qualitative rules from "
        "the knowledge base.\n\n"
        "Core rules:\n"
        "1. NEVER manually guess combinations. Always accept Optuna's suggested "
        "parameters as your starting point. You may request a fixed override only "
        "when explicitly testing a knowledge-base hypothesis — state the hypothesis.\n"
        "2. Read the knowledge base RULES tier first. Rules represent invariants "
        "validated across 200+ trials and MUST be respected.\n"
        "3. If Optuna suggests a combination that violates a Rule, propose a "
        "constrained alternative and explain why.\n"
        "4. Output a single JSON object: {\"proposed_hypothesis\": {...params...}, "
        "\"rationale\": \"...\", \"knowledge_rule_applied\": \"...or null\"}.\n"
        "5. Keep rationale under 80 words. Be precise."
    ),

    "creator": (
        "You are the Creator for the Atwater cognitive architecture. "
        "You execute content generation for the parameter combination proposed by "
        "the Director Engine.\n\n"
        "Core rules:\n"
        "1. Read the current_hypothesis from shared state before doing anything.\n"
        "2. Apply knowledge base PATTERNS relevant to the hypothesis parameters.\n"
        "3. After generating, perform a self-critique: assess the output against "
        "known patterns. Be honest — do not inflate confidence.\n"
        "4. If you used a novel technique not represented in the knowledge base and "
        "you believe it performed well, set suggest_knowledge_write to true.\n"
        "5. Output JSON: {\"output_path\": \"...\", \"self_critique\": \"...\", "
        "\"suggest_knowledge_write\": bool, \"novel_technique\": \"...or null\"}.\n"
        "6. Self-critique must reference specific knowledge patterns, not generic praise."
    ),

    "grader": (
        "You are the Grader Engine for the Atwater cognitive architecture. "
        "You produce structured quality scores that drive both Optuna optimization "
        "and knowledge base updates.\n\n"
        "Core rules:\n"
        "1. Score every rubric dimension independently. Do NOT let dimensions "
        "anchor each other.\n"
        "2. Every score requires reasoning — minimum one concrete observation per "
        "dimension. Vague reasoning like 'looks good' is a failure.\n"
        "3. Overall score = weighted average per the rubric's dimension weights. "
        "Show your arithmetic.\n"
        "4. If you observe a finding not present in the knowledge base AND the "
        "overall score >= 0.75, set suggest_knowledge_write to true and populate "
        "novel_finding with a precise, falsifiable statement.\n"
        "5. Output JSON matching the schema: {\"trial_id\": int, "
        "\"overall_score\": float, \"dimensions\": {dim: {\"score\": float, "
        "\"reasoning\": str}}, \"novel_finding\": str|null, "
        "\"suggest_knowledge_write\": bool}."
    ),

    "diversity_guard": (
        "You are the Diversity Guard for the Atwater cognitive architecture. "
        "You prevent optimization stagnation by monitoring asset concentration "
        "and triggering exploration.\n\n"
        "Core rules:\n"
        "1. Any asset appearing in >30% of the last 50 trials must be flagged "
        "for rotation. Be strict — 30.0% exactly is a trigger.\n"
        "2. If the current cycle number is a multiple of 50, set forced_exploration "
        "to true regardless of usage stats.\n"
        "3. Output JSON: {\"asset_status\": {asset: {\"usage_pct\": float, "
        "\"status\": \"ok\"|\"warn\"|\"rotate\"}}, \"diversity_alerts\": [str], "
        "\"forced_exploration\": bool}.\n"
        "4. Alerts must be actionable: name the specific asset and its usage "
        "percentage. Generic alerts like 'diversity is low' are not acceptable."
    ),

    "consolidator": (
        "You are the Consolidator for the Atwater cognitive architecture. "
        "You run every N cycles to compact the knowledge base: merging redundant "
        "entries, promoting validated observations, and archiving stale ones.\n\n"
        "Core rules:\n"
        "1. Promotions: observations validated by Optuna across 200+ trials "
        "with consistent direction may be promoted to patterns. Patterns with "
        "statistical backing across 200+ trials may become rules.\n"
        "2. Merges: entries in the same topic cluster with >80% semantic overlap "
        "should be consolidated into a single, more precise entry.\n"
        "3. Archives: entries not referenced in the last 200 cycles and lacking "
        "Optuna evidence should be archived (not deleted).\n"
        "4. Output JSON: {\"promotions\": [{\"id\": str, \"from\": str, \"to\": str, "
        "\"reason\": str}], \"merges\": [{\"ids\": [str], \"merged_content\": str}], "
        "\"archives\": [{\"id\": str, \"reason\": str}]}.\n"
        "5. Be conservative with promotions. One strong piece of evidence is not "
        "enough. Require statistical convergence."
    ),

    "orchestrator": (
        "You are the Orchestrator for the Atwater cognitive architecture. "
        "You are the context assembler and flow controller — not a producer.\n\n"
        "Core rules:\n"
        "1. Never inject your own creative or evaluative judgements into agent "
        "contexts. Your job is accurate, scoped context delivery.\n"
        "2. Filter shared state by role before passing it to any agent.\n"
        "3. Load only tool schemas relevant to the current task.\n"
        "4. Track cycle state and trigger the consolidator every N cycles.\n"
        "5. On any agent failure, log the error and continue with degraded context "
        "— never abort the cycle silently."
    ),
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _format_knowledge(entries: list[dict[str, Any]]) -> str:
    """Render knowledge entries as a compact numbered list."""
    if not entries:
        return "No relevant knowledge entries found."
    lines = []
    for i, entry in enumerate(entries, 1):
        tier = entry.get("tier", "?").upper()
        confidence = entry.get("confidence", 0.0)
        content = entry.get("content", "")
        cluster = entry.get("topic_cluster", "general")
        lines.append(
            f"{i}. [{tier} | conf={confidence:.2f} | cluster={cluster}] {content}"
        )
    return "\n".join(lines)


def _format_optuna(optuna_context: dict[str, Any] | None) -> str:
    """Render Optuna context as a compact summary."""
    if not optuna_context:
        return "No Optuna context available (first cycle or study not yet populated)."
    parts = []
    if "best_params" in optuna_context:
        parts.append(f"Best params so far: {optuna_context['best_params']}")
    if "best_value" in optuna_context:
        parts.append(f"Best score so far: {optuna_context['best_value']:.4f}")
    if "n_trials" in optuna_context:
        parts.append(f"Completed trials: {optuna_context['n_trials']}")
    if "param_importances" in optuna_context:
        imp = optuna_context["param_importances"]
        sorted_imp = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
        parts.append(
            "Parameter importances: "
            + ", ".join(f"{k}={v:.3f}" for k, v in sorted_imp[:5])
        )
    if "suggested_params" in optuna_context:
        parts.append(f"Optuna suggested for this trial: {optuna_context['suggested_params']}")
    if "recent_summary" in optuna_context:
        parts.append(f"Recent trial summary: {optuna_context['recent_summary']}")
    return "\n".join(parts)


def build_director_prompt(
    state: dict[str, Any],
    knowledge: list[dict[str, Any]],
    optuna_context: dict[str, Any] | None,
) -> list[dict[str, str]]:
    """
    Build the full message list for the Director Engine.

    Parameters
    ----------
    state : dict
        Scoped shared state for the director role (current_hypothesis,
        historical_success_rates, optuna_trial_params, etc.).
    knowledge : list[dict]
        Top-k knowledge entries retrieved for this task.
    optuna_context : dict | None
        Optuna study summary (best params, importances, suggested params).

    Returns
    -------
    list[dict[str, str]]
        OpenAI-format message list.
    """
    knowledge_text = _format_knowledge(knowledge)
    optuna_text = _format_optuna(optuna_context)

    current_hypothesis = state.get("current_hypothesis") or "None (first cycle)"
    suggested_params = (optuna_context or {}).get("suggested_params", {})

    user_content = (
        f"## Current State\n"
        f"Current hypothesis: {current_hypothesis}\n\n"
        f"## Optuna Context\n"
        f"{optuna_text}\n\n"
        f"## Knowledge Base (filtered to your role)\n"
        f"{knowledge_text}\n\n"
        f"## Your Task\n"
        f"Optuna has suggested the following parameters for this trial:\n"
        f"{suggested_params}\n\n"
        f"Review the knowledge base RULES and PATTERNS above. If any Rule is "
        f"violated by the suggested parameters, propose a constrained alternative.\n"
        f"Otherwise, accept the suggestion and explain how it aligns with (or "
        f"explores beyond) current knowledge.\n\n"
        f"Output your proposed_hypothesis JSON now."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPTS["director"]},
        {"role": "user", "content": user_content},
    ]


def build_creator_prompt(
    hypothesis: dict[str, Any],
    knowledge: list[dict[str, Any]],
    critique_mode: bool = False,
) -> list[dict[str, str]]:
    """
    Build the full message list for the Creator.

    Parameters
    ----------
    hypothesis : dict
        The proposed hypothesis from the Director (parameter combo to execute).
    knowledge : list[dict]
        Top-k knowledge entries relevant to the hypothesis parameters.
    critique_mode : bool
        If True, the prompt emphasises self-critique over generation (used
        when the creator is asked to review a draft rather than produce fresh).

    Returns
    -------
    list[dict[str, str]]
        OpenAI-format message list.
    """
    knowledge_text = _format_knowledge(knowledge)

    if critique_mode:
        task_section = (
            "## Your Task (Critique Mode)\n"
            "You have already produced an output. Now perform a rigorous "
            "self-critique:\n"
            "1. List which knowledge PATTERNS you applied and whether the output "
            "actually reflects them.\n"
            "2. Identify any deviation from the hypothesis parameters.\n"
            "3. Rate your own output on a 0–1 scale per dimension: "
            "originality, brand_alignment, technical_quality.\n"
            "4. If you used a novel technique that worked well, flag it.\n"
            "Output the critique JSON now."
        )
    else:
        task_section = (
            "## Your Task\n"
            "Execute content generation for the hypothesis below. Steps:\n"
            "1. Identify which knowledge PATTERNS are directly applicable.\n"
            "2. Generate the output, applying those patterns.\n"
            "3. After generation, run self-critique: compare output against "
            "each applied pattern and assess whether the application was faithful.\n"
            "4. If you used a novel approach that you believe worked, set "
            "suggest_knowledge_write: true and describe it precisely.\n"
            "Output your result JSON now."
        )

    user_content = (
        f"## Hypothesis to Execute\n"
        f"{hypothesis}\n\n"
        f"## Knowledge Base (filtered to your role)\n"
        f"{knowledge_text}\n\n"
        f"{task_section}"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPTS["creator"]},
        {"role": "user", "content": user_content},
    ]


def build_grader_prompt(
    output: dict[str, Any],
    rubric: dict[str, Any],
    knowledge: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    Build the full message list for the Grader Engine.

    Parameters
    ----------
    output : dict
        The creator's output dict (contains output_path, self_critique, etc.).
    rubric : dict
        Grading rubric from the knowledge base. Expected shape:
        {"dimensions": {name: {"description": str, "weight": float}},
         "scoring_scale": str}.
    knowledge : list[dict]
        Top-k knowledge entries relevant to grading this type of output.

    Returns
    -------
    list[dict[str, str]]
        OpenAI-format message list.
    """
    knowledge_text = _format_knowledge(knowledge)

    # Format rubric dimensions.
    dimensions = rubric.get("dimensions", {})
    if dimensions:
        rubric_lines = []
        for dim, spec in dimensions.items():
            desc = spec.get("description", "")
            weight = spec.get("weight", 1.0)
            rubric_lines.append(f"  - {dim} (weight={weight:.2f}): {desc}")
        rubric_text = "Rubric dimensions:\n" + "\n".join(rubric_lines)
        scoring_scale = rubric.get("scoring_scale", "0.0 (worst) to 1.0 (best)")
        rubric_text += f"\nScoring scale: {scoring_scale}"
    else:
        rubric_text = (
            "No rubric provided. Use default dimensions:\n"
            "  - originality (weight=0.33): Novelty relative to recent trials.\n"
            "  - brand_alignment (weight=0.33): Fit with brand guidelines.\n"
            "  - technical_quality (weight=0.34): Execution quality, precision.\n"
            "Scoring scale: 0.0 (worst) to 1.0 (best)."
        )

    output_path = output.get("output_path", "[path not provided]")
    self_critique = output.get("self_critique", "[no self-critique provided]")

    user_content = (
        f"## Output to Grade\n"
        f"Output path: {output_path}\n"
        f"Creator self-critique: {self_critique}\n\n"
        f"## Grading Rubric\n"
        f"{rubric_text}\n\n"
        f"## Knowledge Base (filtered to your role)\n"
        f"{knowledge_text}\n\n"
        f"## Your Task\n"
        f"Score this output against the rubric. Rules:\n"
        f"1. Score each dimension independently — do not let one dimension "
        f"anchor another.\n"
        f"2. Provide one concrete, specific observation per dimension as reasoning.\n"
        f"3. Compute overall_score as the weighted average. Show your arithmetic "
        f"in a 'score_arithmetic' field.\n"
        f"4. If you identify a finding not in the knowledge base AND "
        f"overall_score >= 0.75, set suggest_knowledge_write: true and write a "
        f"precise, falsifiable novel_finding statement.\n"
        f"Output the grading JSON now."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPTS["grader"]},
        {"role": "user", "content": user_content},
    ]


def build_consolidator_prompt(
    importances: dict[str, float],
    cluster_entries: dict[str, list[dict[str, Any]]],
) -> list[dict[str, str]]:
    """
    Build the full message list for the Consolidator.

    Parameters
    ----------
    importances : dict[str, float]
        Optuna parameter importance scores (param_name → importance value).
    cluster_entries : dict[str, list[dict]]
        Knowledge entries grouped by topic cluster. Each entry is a dict
        representation of a KnowledgeEntry.

    Returns
    -------
    list[dict[str, str]]
        OpenAI-format message list.
    """
    # Format importances.
    if importances:
        sorted_imp = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
        imp_lines = [f"  {k}: {v:.4f}" for k, v in sorted_imp]
        importance_text = "Optuna parameter importances:\n" + "\n".join(imp_lines)
    else:
        importance_text = "No parameter importances available yet."

    # Format cluster entries — summarise each cluster.
    cluster_sections = []
    for cluster_name, entries in cluster_entries.items():
        cluster_sections.append(f"### Cluster: {cluster_name} ({len(entries)} entries)")
        for entry in entries:
            eid = entry.get("id", "?")
            tier = entry.get("tier", "?")
            confidence = entry.get("confidence", 0.0)
            validation_count = entry.get("validation_count", 0)
            last_validated = entry.get("last_validated_cycle", 0)
            content = entry.get("content", "")
            optuna_evidence = entry.get("optuna_evidence", {})
            evidence_str = (
                f"optuna_evidence={optuna_evidence}" if optuna_evidence else "no optuna evidence"
            )
            cluster_sections.append(
                f"  [{eid}] tier={tier} | conf={confidence:.2f} | "
                f"validations={validation_count} | last_validated=cycle{last_validated} | "
                f"{evidence_str}\n"
                f"  Content: {content}"
            )
        cluster_sections.append("")

    clusters_text = "\n".join(cluster_sections) if cluster_sections else "No clusters to process."

    user_content = (
        f"## Statistical Context\n"
        f"{importance_text}\n\n"
        f"## Knowledge Base Clusters for Compaction\n"
        f"{clusters_text}\n"
        f"## Your Task\n"
        f"Perform knowledge compaction on the clusters above. For each action:\n\n"
        f"**Promotions** — Eligible if:\n"
        f"  - Observation → Pattern: consistent direction across 200+ trials "
        f"with Optuna evidence, validation_count >= 5.\n"
        f"  - Pattern → Rule: validated across 200+ trials, high Optuna importance "
        f"(> 0.10), validation_count >= 20.\n"
        f"  Be conservative. Require convergent evidence, not a single strong trial.\n\n"
        f"**Merges** — Eligible if:\n"
        f"  - Two or more entries in the same cluster state overlapping facts.\n"
        f"  - Write a single merged entry that is MORE precise than either original.\n\n"
        f"**Archives** — Eligible if:\n"
        f"  - Entry has no Optuna evidence AND last_validated_cycle was more than "
        f"200 cycles ago AND validation_count <= 1.\n"
        f"  - Archive, do not delete.\n\n"
        f"Output the compaction JSON now."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPTS["consolidator"]},
        {"role": "user", "content": user_content},
    ]
