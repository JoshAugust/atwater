# OPTUNA ADVANCED RESEARCH
_Generated: 2026-04-22 | Atwater Project_

---

## EXECUTIVE SUMMARY

Optuna (currently v4.x, confirmed stable at 4.8.0) has several advanced capabilities that are highly relevant to Atwater's cognitive agent architecture. Key findings: **AutoSampler** is a must-adopt, **GPSampler beats TPE for mixed integer problems**, **JournalStorage beats SQLite for concurrent writes**, and **artifacts module** enables storing large outputs per trial (critical for creative generation results).

---

## 1. SAMPLER SELECTION — CRITICAL FINDINGS

### AutoSampler (OptunaHub) — **ADOPT IMMEDIATELY**

The biggest practical finding: Optuna now has an **AutoSampler** via OptunaHub that automatically selects the optimal algorithm based on problem characteristics. This outperforms always using TPE.

```python
import optunahub
import optuna

study = optuna.create_study(
    sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler()
)
```

**Install:**
```bash
pip install optunahub
pip install -r https://hub.optuna.org/samplers/auto_sampler/requirements.txt
```

**AutoSampler selection logic:**
- **GPSampler** during early stages (excellent sample efficiency, best for <200-300 evaluations)
- **TPESampler** for problems with categorical variables
- Switches dynamically based on trial count and search space shape

### GPSampler vs TPE vs CMA-ES

| Sampler | Best For | Weakness |
|---------|----------|----------|
| **GPSampler** | Low-to-medium trial counts (<300), mixed int/float | Slow at scale, no categoricals |
| **TPESampler** | Categorical params, high trial counts | Less sample-efficient early on |
| **CmaEsSampler** | Continuous params, no categoricals | Breaks with mixed types |
| **NSGAIISampler** | Multi-objective | Requires many trials |
| **GPSampler** | Integer variables | Better than CMA-ES for int |

**Benchmark insight:** On 8-dimensional Styblinski-Tang function with integer variables, GPSampler significantly outperforms CMA-ES. On purely continuous problems, they're comparable.

### Multi-Objective for Creative Tasks

Use `NSGAIISampler` or `NSGAIIISampler` for multi-objective optimization:

```python
study = optuna.create_study(
    directions=["maximize", "maximize"],  # e.g., quality + diversity
    sampler=optuna.samplers.NSGAIISampler()
)

# Access Pareto front
pareto_trials = study.best_trials
for trial in pareto_trials:
    print(trial.values)  # [quality_score, diversity_score]
```

**For Atwater creative tasks:** Optimize simultaneously for:
- `quality` (LLM judge score)
- `diversity` (embedding distance from prior outputs)
- `style_adherence` (cosine similarity to target aesthetic)

---

## 2. PRUNING STRATEGIES

Available pruners (current stable):
- **MedianPruner** — stop if below median at intermediate step (default, good baseline)
- **SuccessiveHalvingPruner** — Hyperband-style, best for iterative evaluations
- **HyperbandPruner** — Full Hyperband algorithm
- **ThresholdPruner** — Stop if value crosses absolute threshold
- **WilcoxonPruner** — Statistical test-based pruning
- **PatientPruner** — Adds tolerance before pruning (wraps another pruner)

**Key constraint: Pruners only work with single-objective optimization.**

For Atwater's expensive LLM evaluations:
```python
# ThresholdPruner for creative tasks — prune if score is clearly terrible
pruner = optuna.pruners.ThresholdPruner(lower=0.3)  # below 0.3 is hopeless

# Or: PatientPruner wrapping Median — don't prune too early
pruner = optuna.pruners.PatientPruner(
    optuna.pruners.MedianPruner(), patience=3
)
```

**Usage pattern in objective:**
```python
def objective(trial):
    # ... generate some intermediate results ...
    intermediate_score = quick_eval()
    trial.report(intermediate_score, step=0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    # ... expensive full evaluation ...
    return final_score
```

---

## 3. DISTRIBUTED OPTIMIZATION

Three tiers of parallelism:

### Multi-Thread (same process)
```python
storage = JournalStorage(JournalFileBackend("./journal.log"))
study = optuna.create_study(storage=storage)
study.optimize(objective, n_trials=100, n_jobs=4)
```
Note: Currently GIL-limited in Python <3.14. Python 3.14 removes GIL = real speedup.

### Multi-Process (same machine)
```python
# Process 1:
storage = "sqlite:///study.db"  # or JournalStorage with file backend
study = optuna.create_study(study_name="atwater", storage=storage, load_if_exists=True)
study.optimize(objective, n_trials=25)

# Process 2 (separate script/process, same storage):
study = optuna.load_study(study_name="atwater", storage=storage)
study.optimize(objective, n_trials=25)
```

### Multi-Node
Use `RDBStorage` (PostgreSQL recommended) or `GrpcStorageProxy` for thousands of nodes.

**CRITICAL: SQLite Performance Warning**
- SQLite with RDBStorage suffers from write contention at scale
- For >1 concurrent worker: use **JournalStorage** instead
- JournalStorage with file backend is append-only → no write locks
- SQLite OK for sequential single-worker runs up to 10K+ trials

```python
# Better than RDB SQLite for concurrent workers:
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

storage = JournalStorage(JournalFileBackend("./optuna_journal.log"))
```

---

## 4. CUSTOM SAMPLER (DOMAIN-SPECIFIC EXPLORATION)

Optuna supports fully custom samplers via `BaseSampler`. Required methods:
- `infer_relative_search_space(study, trial)` → dict
- `sample_relative(study, trial, search_space)` → dict of param values
- `sample_independent(study, trial, param_name, param_distribution)` → single value

**Example: Simulated Annealing Sampler pattern:**
```python
class AtwaterDomainSampler(optuna.samplers.BaseSampler):
    def __init__(self, temperature=100, seed=None):
        self._rng = np.random.RandomState(seed)
        self._temperature = temperature
        self._current_trial = None

    def sample_relative(self, study, trial, search_space):
        if not search_space:
            return {}
        # Custom logic: prefer parameters close to best known params
        # but with creative exploration bias
        prev_trial = study.best_trial if study.best_trial else None
        # ... neighborhood sampling ...
        return params

    def sample_independent(self, study, trial, param_name, param_distribution):
        return optuna.samplers.RandomSampler().sample_independent(
            study, trial, param_name, param_distribution
        )
```

**For Atwater:** A custom sampler could embed Atwater's aesthetic knowledge directly into the exploration strategy — e.g., biasing toward parameter regions that correlate with high embedding similarity to target style profiles.

---

## 5. WARM START / PRIOR KNOWLEDGE

### Enqueue Known-Good Trials
```python
study = optuna.create_study(storage=storage, load_if_exists=True)

# Seed with prior knowledge before optimizing
study.enqueue_trial({"temperature": 0.7, "top_p": 0.9, "style_weight": 0.8})
study.enqueue_trial({"temperature": 0.9, "top_p": 0.95, "style_weight": 0.6})

# These run first, then Bayesian optimization takes over
study.optimize(objective, n_trials=100)
```

### Resume Studies (Warm Start from Previous Run)
```python
# Resuming preserves all trial history — sampler adapts from existing data
study = optuna.create_study(
    study_name="atwater_style_opt",
    storage="sqlite:///atwater.db",
    load_if_exists=True  # KEY: loads all prior trials
)
study.optimize(objective, n_trials=50)  # Continues from where it left off
```

**Note:** Sampler state is NOT stored in the DB. To resume with same sampler state (for reproducibility), pickle the sampler separately.

### Re-using Best Trial for New Context
```python
# After optimization, deploy best params to a different context
best = study.best_trial
new_study.enqueue_trial(best.params)  # Seed new study with transfer knowledge
```

---

## 6. ARTIFACTS MODULE (Critical for Atwater)

Optuna v3.3+ has an artifact module for storing large per-trial outputs:

```python
from optuna.artifacts import upload_artifact, FileSystemArtifactStore

artifact_store = FileSystemArtifactStore("./artifacts/")

def objective(trial):
    params = {
        "temperature": trial.suggest_float("temperature", 0.1, 2.0),
        "style_weight": trial.suggest_float("style_weight", 0.0, 1.0),
    }
    # Generate creative output
    output = generate_with_llm(params)
    
    # Save the actual generated text as artifact
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(output)
        artifact_id = upload_artifact(trial, f.name, artifact_store)
    
    score = evaluate(output)
    return score
```

**Why this matters for Atwater:** Store actual generated outputs (images, text, audio) per trial. optuna-dashboard can display/download them via web UI.

---

## 7. VISUALIZATION & DASHBOARD

```bash
pip install optuna-dashboard
optuna-dashboard sqlite:///atwater.db
```

Key visualizations available:
- Optimization history
- Hyperparameter importance (via `optuna.visualization.plot_param_importances`)
- Pareto front (for multi-objective)
- Parameter relationships

**Programmatic importance:**
```python
importances = optuna.importance.get_param_importances(study)
# Returns dict of {param_name: importance_score}
```

---

## 8. REINFORCEMENT LEARNING INTEGRATION

Optuna integrates with RL via:
- **Optuna + Stable Baselines 3**: optimize RL hyperparameters (lr, gamma, batch_size)
- **RL-as-sampler**: Use RL agent as a custom sampler (advanced pattern)
- **OptunaHub**: Community samplers include RL-based exploration

Pattern for optimizing RL agent hyperparams:
```python
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)
    n_steps = trial.suggest_int("n_steps", 64, 2048, log=True)
    
    model = PPO("MlpPolicy", env, learning_rate=lr, gamma=gamma, n_steps=n_steps)
    model.learn(total_timesteps=10000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    return mean_reward
```

---

## 9. WHAT TO CHANGE IN ATWATER's OPTUNA CODE

### Immediate Changes:

1. **Switch to AutoSampler** — stop hardcoding TPE, let AutoSampler pick
   ```python
   pip install optunahub
   sampler = optunahub.load_module("samplers/auto_sampler").AutoSampler()
   ```

2. **Switch to JournalStorage** if using concurrent workers (replace SQLite RDB)
   ```python
   storage = JournalStorage(JournalFileBackend("./journal.log"))
   ```

3. **Add enqueue_trial for warm starts** — seed each new study with best params from previous runs

4. **Multi-objective for creative tasks** — if optimizing quality AND diversity simultaneously, use NSGAIISampler with `directions=["maximize", "maximize"]`

5. **Add artifact storage** for creative outputs — store actual generated text/content alongside scores

6. **Use PatientPruner** wrapping MedianPruner for expensive LLM evaluations — avoids pruning too eagerly

### Architecture Pattern for Atwater:
```python
# Recommended Atwater optimization setup
import optuna
import optunahub
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

def create_atwater_study(study_name: str, direction: str = "maximize"):
    storage = JournalStorage(JournalFileBackend(f"./studies/{study_name}.log"))
    
    try:
        sampler = optunahub.load_module("samplers/auto_sampler").AutoSampler()
    except Exception:
        sampler = optuna.samplers.TPESampler(seed=42)  # fallback
    
    pruner = optuna.pruners.PatientPruner(
        optuna.pruners.MedianPruner(n_startup_trials=5),
        patience=2
    )
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction=direction,
        load_if_exists=True
    )
    return study
```

---

## SOURCES

- Optuna docs (stable, v4.8.0): https://optuna.readthedocs.io/en/stable/
- AutoSampler blog: https://medium.com/optuna/autosampler-automatic-selection-of-optimization-algorithms-in-optuna-1443875fd8f9
- Optuna Artifacts tutorial: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/012_artifact_tutorial.html
- Distributed optimization: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
- User-defined sampler: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/005_user_defined_sampler.html
