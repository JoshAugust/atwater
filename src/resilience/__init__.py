"""
src.resilience — Robustness and error-handling subsystems for Atwater.

Provides:
  - CircuitBreaker     : Fail-fast wrapper for LLM/external calls.
  - CircuitOpenError   : Raised when the circuit is open.
  - RateLimiter        : Token-bucket rate limiter (sync + async).
  - CheckpointManager  : Cycle-level checkpointing with auto-rotation.
  - CheckpointData     : NamedTuple returned from load_checkpoint().
  - HealthChecker      : System health verification across components.
  - HealthResult       : NamedTuple for individual component health.
  - ShutdownHandler    : POSIX signal handling with callback registry.
  - TFIDFFallback      : TF-IDF-based knowledge retrieval fallback.
  - MockLLMFallback    : Sensible defaults when LM Studio is unavailable.
  - EmbeddingFallback  : TF-IDF vectors as embedding model replacement.
"""

from src.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError
from src.resilience.checkpointing import CheckpointData, CheckpointManager
from src.resilience.fallbacks import EmbeddingFallback, MockLLMFallback, TFIDFFallback
from src.resilience.graceful_shutdown import ShutdownHandler
from src.resilience.health_check import HealthChecker, HealthResult
from src.resilience.rate_limiter import RateLimiter

__all__ = [
    "CircuitBreaker",
    "CircuitOpenError",
    "RateLimiter",
    "CheckpointManager",
    "CheckpointData",
    "HealthChecker",
    "HealthResult",
    "ShutdownHandler",
    "TFIDFFallback",
    "MockLLMFallback",
    "EmbeddingFallback",
]
