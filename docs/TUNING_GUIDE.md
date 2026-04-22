# Tuning Guide

How to configure Atwater for your specific domain and creative objectives.

---

## Defining Your Search Space

The search space is the set of all parameters Optuna optimises over. It lives in `src/optimization/trial_adapter.py` as a `SearchSpace` dataclass with three dimension types:

```python
SearchSpace(
    categorical={...},   # Discrete choices (background style, layout type)
    continuous={...},     # Float ranges (opacity, font scale)
    integer={...},        # Int ranges with optional step (font size)
)
```

### Design principles

1. **Start small.** 4-6 categorical dimensions + 2-3 continuous is plenty for the first 500 trials. Optuna needs ~10× the number of dimensions in completed trials before TPE becomes effective.

2. **Make dimensions independent.** If "gradient background" always requires "dark text", that's a constraint for the Creator, not a search dimension. Correlated dimensions waste trials exploring impossible combinations.

3. **Use meaningful categories.** Don't enumerate 200 background images as categories. Group them: "dark", "gradient", "minimal", "textured". The system will learn which groups work; you can sub-divide later.

4. **Continuous dimensions need reasonable bounds.** Optuna searches the full range. If `opacity` ranges from 0.0 to 1.0 but anything below 0.3 looks terrible, set the bound to (0.3, 1.0).

### Example: E-commerce product ads

```python
ECOMMERCE_SEARCH_SPACE = SearchSpace(
    categorical={
        "background": ["lifestyle", "studio-white", "gradient", "contextual", "abstract"],
        "layout": ["hero-product", "split-50/50", "grid-4up", "story-scroll", "minimal"],
        "product_angle": ["front", "45-degree", "overhead", "in-use", "detail"],
        "cta_style": ["button-solid", "button-outline", "text-link", "floating"],
        "color_mood": ["warm", "cool", "neutral", "bold", "muted"],
    },
    continuous={
        "product_scale": (0.3, 0.9),        # how much of the frame the product fills
        "text_contrast_ratio": (4.5, 12.0),  # WCAG AA minimum is 4.5:1
        "whitespace_ratio": (0.1, 0.4),      # % of canvas that's breathing room
    },
    integer={
        "headline_font_size": (18, 64, 2),
    },
)
```

---

## Choosing Evaluation Metrics

Atwater's grader produces a structured score. The overall_score is what Optuna optimises. The dimension scores are qualitative — they feed the knowledge base.

### Single-objective (default)

One number between 0.0 and 1.0. Simpler, faster convergence.

```python
study = create_study(name="my-study", direction="maximize")
```

### Multi-objective (quality + diversity)

Returns two scores; Optuna finds the Pareto front.

```python
study = create_multi_objective_study(
    name="my-study",
    directions=["maximize", "maximize"],
)
```

Use multi-objective when:
- You want creative variety, not just the single best template
- Your audience is segmented (different styles for different markets)
- You're building a portfolio of options

### Dimension examples

| Dimension | What It Measures | When to Weight Heavily |
|-----------|-----------------|----------------------|
| **originality** | How novel is this compared to recent outputs | Early exploration phase |
| **brand_alignment** | Does it follow brand guidelines | Always — this is a hard constraint |
| **technical_quality** | Image quality, text legibility, layout balance | Production phase |
| **emotional_impact** | Does it evoke the intended feeling | Brand-focused campaigns |
| **conversion_potential** | Would someone click/buy from this | Performance marketing |

---

## Setting Grading Rubrics

Rubrics are knowledge base entries at the "rule" tier. Seed them before your first real run:

```python
from src.memory import KnowledgeBase

kb = KnowledgeBase(db_path="knowledge.db")

# Hard brand constraints (rule tier, high confidence)
kb.knowledge_write(
    "Brand colours must be from the approved palette: #1A1A2E, #16213E, #0F3460, #E94560. "
    "Any creative using off-palette colours as primary scores 0 for brand_alignment.",
    tier="rule", confidence=1.0, topic_cluster="brand"
)

# Typography rules
kb.knowledge_write(
    "Headlines: Inter Bold 600 or Satoshi Black. Body: Inter Regular 400. "
    "Never use more than 2 typefaces per creative. Minimum 4.5:1 contrast ratio.",
    tier="rule", confidence=1.0, topic_cluster="typography"
)

# Layout heuristics (pattern tier — softer, learned over time)
kb.knowledge_write(
    "Product-first layouts (hero-product, minimal) score 15-20% higher for "
    "conversion_potential than lifestyle or contextual backgrounds.",
    tier="pattern", confidence=0.7, topic_cluster="layout"
)
```

The Grader agent retrieves relevant rules via semantic search before scoring. More specific rules = more consistent grading.

---

## Tuning Temperature Schedules

Temperature controls exploration vs exploitation across cycles. Configured in `src/learning/temperature_schedule.py`.

### Default schedule

```
Cycles 1-50:    T = 0.8  (high exploration)
Cycles 50-200:  T = 0.8 → 0.3  (linear annealing)
Cycles 200+:    T = 0.3  (exploitation)
```

### When to adjust

- **Increase start temperature** if early cycles are too conservative (not exploring enough of the search space)
- **Extend annealing period** if you have a large search space (many dimensions × many categories)
- **Lower floor temperature** if you want to converge faster on the optimum
- **Add periodic reheating** if you suspect the system is stuck in a local optimum — temporarily raise T every N cycles

### Custom schedule

```python
from src.learning.temperature_schedule import TemperatureSchedule

schedule = TemperatureSchedule(
    initial_temp=0.9,
    final_temp=0.2,
    anneal_start=100,    # start annealing later
    anneal_end=500,      # anneal over more cycles
    reheat_interval=200, # reheat every 200 cycles
    reheat_temp=0.6,     # reheat to this temperature
)
```

---

## When to Switch from Single to Multi-Objective

**Start with single-objective.** It's simpler, converges faster, and gives you a clear "best" result.

Switch to multi-objective when:

1. **You need diverse outputs** — a single best template is not enough; you want a portfolio
2. **Objectives genuinely conflict** — maximising quality might reduce diversity; you need the Pareto front to see the tradeoff
3. **You have 200+ trials** of single-objective data — enough to understand the landscape before adding complexity

The switch is mechanical:

```python
# Before (single)
study = create_study(name="my-study", direction="maximize")

# After (multi)
study = create_multi_objective_study(
    name="my-study",
    directions=["maximize", "maximize"],  # quality, diversity
)
```

The Grader must return a list of scores instead of a scalar. `study.best_trials` gives the Pareto front.

---

## Interpreting Optuna Parameter Importances

After 50+ completed trials, Optuna can tell you which parameters matter most:

```python
from src.optimization import get_importances

importances = get_importances(study)
# {"background": 0.35, "layout": 0.25, "product_angle": 0.15, ...}
```

### What the numbers mean

- **> 0.3**: This parameter has a huge impact. Spend time curating its categories.
- **0.1–0.3**: Meaningful but not dominant. Keep it in the search space.
- **< 0.05**: This parameter barely matters. Consider removing it or fixing it to its best value (fewer dimensions = faster convergence).

### Action items by importance

| Importance | Action |
|-----------|--------|
| Top parameter (>0.3) | Sub-divide its categories for finer-grained optimisation |
| Moderate (0.1-0.3) | Keep as-is. Check if any single category dominates. |
| Low (<0.05) | Fix to best value, remove from search space, or merge with another dimension |

---

## Reading the Knowledge Base

The knowledge base contains three tiers of insight:

### Rules (permanent, high confidence)
```python
entries = kb.knowledge_read("font choice", tier="rule")
```
These are hard constraints validated across 200+ trials. Treat them as ground truth.

### Patterns (heuristics, medium confidence)
```python
entries = kb.knowledge_read("dark backgrounds", tier="pattern")
```
Reliable heuristics re-validated every 100 cycles. May have exceptions.

### Observations (single-cycle findings, low confidence)
```python
entries = kb.knowledge_read("gradient overlay", tier="observation")
```
Individual findings. Auto-archived after 50 cycles unless promoted. Useful for debugging — "what did the system learn on cycle 347?"

### Browsing all entries

```python
store = KnowledgeStore(db_path="knowledge_store.db")

# All active rules
rules = store.list_entries(tier="rule")

# Everything in the "typography" cluster
typo = store.list_entries(topic_cluster="typography")

# Including archived (for history)
all_entries = store.list_entries(include_archived=True)
```

### What to look for

- **Contradictions**: Entries with non-empty `contradicted_by` lists. The Consolidator should resolve these, but check manually after long runs.
- **Stale patterns**: Patterns with `last_validated_cycle` far behind the current cycle. May need manual re-evaluation.
- **Observation clusters**: Many observations in the same topic_cluster suggest a pattern waiting to be promoted.
