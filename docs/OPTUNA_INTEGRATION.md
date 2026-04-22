# Optuna Integration Guide

How to wire Optuna as the statistical backbone for the cognitive architecture.

---

## Why Optuna

When you have a fixed pool of assets across N dimensions (X backgrounds, Y layouts, Z product shots) with variable parameters for each, you need:

1. **Tracking** what combinations have been tried
2. **Scoring** which combos work best
3. **Steering** toward promising regions (not random exploration)
4. **Statistical queries** on what matters most

Optuna handles all four. Your agents stop tracking combos in natural language and start querying actual numbers.

---

## Setup

```python
import optuna
from optuna.samplers import TPESampler

# Create persistent study (survives restarts)
study = optuna.create_study(
    study_name="caffy_production",
    direction="maximize",
    storage="sqlite:///optuna_trials.db",
    load_if_exists=True,
    sampler=TPESampler(
        seed=42,
        n_startup_trials=20,  # Random exploration for first 20 trials
        multivariate=True,    # Model parameter interactions
    )
)
```

---

## Defining the Search Space

```python
# Asset pools (your N dimensions)
BACKGROUNDS = ["dark", "gradient", "minimal", "textured", "abstract"]
LAYOUTS = ["hero", "split", "grid", "asymmetric", "stacked"]
PRODUCT_SHOTS = ["front", "angle", "lifestyle", "closeup", "context"]
TYPOGRAPHY = ["sans-modern", "sans-classic", "serif-editorial", "mono"]

def suggest_params(trial):
    """Called by director_engine to pick next combination."""
    params = {
        # Categorical dimensions (fixed pools)
        "background": trial.suggest_categorical("background", BACKGROUNDS),
        "layout": trial.suggest_categorical("layout", LAYOUTS),
        "product_shot": trial.suggest_categorical("shot", PRODUCT_SHOTS),
        "typography": trial.suggest_categorical("typography", TYPOGRAPHY),
        
        # Continuous parameters (variable)
        "bg_opacity": trial.suggest_float("bg_opacity", 0.2, 1.0),
        "font_scale": trial.suggest_float("font_scale", 0.8, 1.5),
        "padding_ratio": trial.suggest_float("padding_ratio", 0.02, 0.15),
        "contrast_ratio": trial.suggest_float("contrast_ratio", 3.0, 10.0),
        
        # Integer parameters
        "hero_font_size": trial.suggest_int("hero_font_size", 24, 72, step=4),
    }
    return params
```

### Adding New Dimensions

Just add more `trial.suggest_*()` calls. Optuna handles the expanded space automatically. Existing trial history remains valid — new dimensions start exploring from scratch while learned dimensions keep their optimization.

---

## Agent Pipeline Integration

### Full Cycle

```python
def run_cycle(study):
    # 1. Director suggests params
    trial = study.ask()
    params = suggest_params(trial)
    
    # 2. Write to shared state
    state_write("current_hypothesis", params)
    state_write("current_trial_id", trial.number)
    
    # 3. Creator generates output
    output = creator_generate(params)
    
    # 4. Grader evaluates
    score = grader_evaluate(output, params)
    
    # 5. Report back to Optuna
    study.tell(trial, score)
    
    # 6. Check diversity
    diversity_check(study, threshold=0.30)
    
    return score
```

### Director Engine Integration

```python
def director_decide(study, knowledge_base):
    """Director uses Optuna + knowledge to pick next combo."""
    
    # Check if knowledge base has a hypothesis to test
    hypothesis = knowledge_base.read("untested_hypotheses", tier="observation")
    
    if hypothesis and random.random() < 0.2:  # 20% of time, test knowledge hypotheses
        trial = study.ask()
        # Override Optuna's suggestion with the hypothesis
        fixed_params = hypothesis.suggested_params
        study.tell(trial, run_with_fixed_params(fixed_params))
    else:
        # Let Optuna's Bayesian optimization steer
        trial = study.ask()
        params = suggest_params(trial)
        score = run_pipeline(params)
        study.tell(trial, score)
```

### Diversity Guard Integration

```python
def diversity_check(study, threshold=0.30):
    """Flag overused assets and force exploration."""
    df = study.trials_dataframe()
    recent = df.tail(50)  # Last 50 trials
    
    for dim in ["params_background", "params_layout", "params_shot"]:
        usage = recent[dim].value_counts(normalize=True)
        overused = usage[usage > threshold]
        
        if not overused.empty:
            for asset in overused.index:
                state_write(f"deprecated_{dim}_{asset}", True)
                # Optuna will naturally avoid if we add a constraint
                print(f"⚠️ {asset} used in {usage[asset]:.0%} of recent trials")
    
    # Force random exploration every 50 cycles
    if len(df) % 50 == 0:
        trial = study.ask()
        # Use RandomSampler for this one trial
        random_params = {
            "background": random.choice(BACKGROUNDS),
            "layout": random.choice(LAYOUTS),
            "shot": random.choice(PRODUCT_SHOTS),
            # ... etc
        }
        score = run_with_fixed_params(random_params)
        study.tell(trial, score)
```

---

## Statistical Queries

### Parameter Importance (which dimensions matter most)

```python
importances = optuna.importance.get_param_importances(study)
# Returns: {"background": 0.35, "layout": 0.28, "font_scale": 0.15, ...}
```

### Best Parameters

```python
study.best_params
# {"background": "dark", "layout": "hero", "shot": "lifestyle", ...}

study.best_value  # 0.92
study.best_trial  # Full trial object with all metadata
```

### Per-Dimension Analysis

```python
df = study.trials_dataframe()

# Average score per background type
df.groupby("params_background")["value"].agg(["mean", "std", "count"])

# Best layout for each background
df.groupby(["params_background", "params_layout"])["value"].mean().unstack()

# Score trend over time
df["value"].rolling(20).mean().plot()
```

### Combination Heatmaps

```python
import seaborn as sns
pivot = df.pivot_table(
    values="value",
    index="params_background",
    columns="params_layout",
    aggfunc="mean"
)
sns.heatmap(pivot, annot=True, fmt=".2f")
```

---

## Multi-Objective Optimization

If you need to optimise for multiple metrics simultaneously (e.g. engagement AND originality AND brand alignment):

```python
study = optuna.create_study(
    directions=["maximize", "maximize", "maximize"],
    storage="sqlite:///trials.db"
)

def objective(trial):
    params = suggest_params(trial)
    output = creator_generate(params)
    scores = grader_evaluate_multi(output)
    return scores["engagement"], scores["originality"], scores["brand_alignment"]

# Pareto front: best trade-offs
study.best_trials  # Returns all Pareto-optimal trials
```

---

## Bridge to Knowledge Base

The consolidation agent reads Optuna stats and writes qualitative interpretations:

```python
def consolidation_pass(study, knowledge_base, current_cycle):
    """Run every 50 cycles. Reads stats, writes insights."""
    
    importances = optuna.importance.get_param_importances(study)
    df = study.trials_dataframe()
    
    # Find statistically significant findings
    for dim in importances:
        if importances[dim] > 0.1:  # Meaningful impact
            dim_col = f"params_{dim.replace('params_', '')}"
            group_stats = df.groupby(dim_col)["value"].agg(["mean", "std", "count"])
            
            best = group_stats["mean"].idxmax()
            worst = group_stats["mean"].idxmin()
            
            if group_stats.loc[best, "count"] >= 20:  # Enough evidence
                knowledge_base.write(
                    content=f"{best} outperforms {worst} by {group_stats.loc[best, 'mean'] - group_stats.loc[worst, 'mean']:.1%} (n={group_stats.loc[best, 'count']})",
                    tier="pattern" if group_stats.loc[best, "count"] >= 50 else "observation",
                    confidence=min(0.95, group_stats.loc[best, "count"] / 100),
                    topic_cluster=dim,
                    optuna_evidence={
                        "param": dim,
                        "best_value": best,
                        "mean_score": float(group_stats.loc[best, "mean"]),
                        "sample_size": int(group_stats.loc[best, "count"]),
                    }
                )
```

This is the key bridge: **Optuna holds truth, knowledge base holds interpretation.** Agents read interpretations during generation, but the interpretations are backed by real numbers.

---

## Exploration vs Exploitation Schedule

```python
def get_sampler(total_trials):
    """Shift from exploration to exploitation over time."""
    if total_trials < 50:
        # Pure exploration
        return optuna.samplers.RandomSampler(seed=42)
    elif total_trials < 200:
        # Balanced (default TPE)
        return optuna.samplers.TPESampler(seed=42, n_startup_trials=0)
    else:
        # Heavy exploitation with periodic exploration
        return optuna.samplers.TPESampler(
            seed=42,
            n_startup_trials=0,
            consider_endpoints=True,  # More precise near optima
        )
```
