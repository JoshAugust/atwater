# 16-Hour Build Plan (Revised with Optuna)

## Hours 1-3: Optuna Integration as Statistical Backbone
- Define full search space (all N dimensions + continuous parameters)
- SQLite-backed study for concurrent agent access
- Wire director_engine → trial.suggest, grader_engine → trial.report
- Build adapter layer: shared memory keys map to trial params
- Replaces most of original "knowledge graph prototype"

## Hours 4-6: Two-Tier Knowledge Split
- **Quantitative layer** = Optuna's trial DB (raw numbers, importances, scores)
- **Qualitative layer** = hierarchical knowledge base (interpretations, creative insights)
- Bridge: consolidation agent reads Optuna stats → writes human-readable takeaways
- Statistical truth feeds qualitative memory, not the reverse

## Hours 7-10: Consolidation Agent (Lean Version)
- Focus on creative knowledge only (Optuna handles statistical contradictions)
- Hierarchical tiers: Rules → Patterns → Observations
- Promotion criteria includes Optuna evidence ("promoted because p<0.05 across 200+ trials")
- Confidence decay + automatic tier demotion
- Topic clustering via HDBSCAN

## Hours 11-13: Intelligent Exploration vs Exploitation
- Tune TPE sampler for use case (early exploration → late exploitation)
- Wire diversity_guard into Optuna trial history
- Asset concentration alerts (>30% of recent trials)
- Creative serendipity: force random trial every 50 cycles
- Custom samplers for constrained exploration

## Hours 14-16: Scale Stress Test
- Simulate 2,000 cycles with synthetic data
- Measure: active KB size (target: under 100 entries), query latency (<100ms), retrieval precision
- Key metric: does cycle 2,000 produce measurably better output than cycle 200?
- Dashboard: Optuna built-in viz + KB health summary

## What Got Cut (vs Original Plan)
- ❌ Knowledge graph prototype → Optuna trial relationships replace this
- ❌ Flat vs hierarchical retrieval benchmarking → quant/qual split makes this moot
- ❌ Forgetting mechanism for statistical findings → Optuna keeps everything, recency weighting handles staleness
