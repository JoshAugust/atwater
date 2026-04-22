"""
src.learning — Advanced exploration and learning modules for Atwater.

Exports
-------
ReflexionEngine     Verbal RL: structured per-cycle reflection seeding next cycle.
Reflection          Dataclass holding a single cycle's reflection output.
StrategySelector    Thompson Sampling multi-armed bandit for strategy selection.
TemperatureScheduler Adaptive cosine annealing with plateau detection.
CollapseDetector    Mode collapse detection from trial diversity.
CollapseAlert       Dataclass for collapse detection output.
"""

from src.learning.reflexion import Reflection, ReflexionEngine
from src.learning.strategy_selector import StrategySelector
from src.learning.temperature_schedule import TemperatureScheduler
from src.learning.collapse_detector import CollapseAlert, CollapseDetector

__all__ = [
    "Reflection",
    "ReflexionEngine",
    "StrategySelector",
    "TemperatureScheduler",
    "CollapseAlert",
    "CollapseDetector",
]
