"""
core/circuit_breaker.py
─────────────────────────────────────────────────────────────────────────────
Per-Provider LLM Circuit Breaker (Phase 3.1)

Implements a state machine to protect downstream LLM providers from being
hammered during prolonged outages.

States:
  CLOSED    — Normal operation. Requests flow through.
  OPEN      — Fast-failure mode. Requests rejected immediately (returns None).
  HALF_OPEN — Probation. Allows 1 request through to test if the provider recovered.

Transitions:
  CLOSED -> OPEN:      On ``failure_threshold`` consecutive failures.
  OPEN -> HALF_OPEN:   After ``recovery_timeout`` seconds have elapsed.
  HALF_OPEN -> OPEN:   On 1 failure.
  HALF_OPEN -> CLOSED: On 1 success.
"""

import time
import logging
from enum import Enum
from typing import Any
from langchain_core.runnables import Runnable, RunnableConfig

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class ProviderCircuitBreaker:
    """A circuit breaker protecting a specific LLM provider."""
    
    def __init__(self, provider_name: str, failure_threshold: int = 3, recovery_timeout: float = 60.0):
        self.provider_name = provider_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0

    def allow_request(self) -> bool:
        """Determines if a request should be allowed through based on the current state."""
        if self.state == CircuitState.CLOSED:
            return True
            
        if self.state == CircuitState.OPEN:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                logger.warning(
                    "[CircuitBreaker] %s timeout expired. Transitioning OPEN -> HALF_OPEN", 
                    self.provider_name
                )
                self.state = CircuitState.HALF_OPEN
                return True
            return False
            
        # HALF_OPEN: Only allow one request (in a synchronous/simple implementation).
        # In a highly concurrent environment, a token bucket or lock would be needed.
        # For LangGraph where nodes execute somewhat sequentially per branch, this is sufficient.
        return True

    def record_success(self) -> None:
        """Record a successful request."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info(
                "[CircuitBreaker] %s request succeeded. Transitioning HALF_OPEN -> CLOSED", 
                self.provider_name
            )
            self.state = CircuitState.CLOSED
        
        # Reset counters on any success
        self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            logger.warning(
                "[CircuitBreaker] %s probe failed. Transitioning HALF_OPEN -> OPEN", 
                self.provider_name
            )
            self.state = CircuitState.OPEN
            return
            
        if self.state == CircuitState.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                logger.error(
                    "[CircuitBreaker] %s failure threshold (%d) reached. Transitioning CLOSED -> OPEN", 
                    self.provider_name, self.failure_threshold
                )
                self.state = CircuitState.OPEN


class CircuitBreakerRunnable(Runnable):
    """A LangChain Runnable wrapper that integrates the Circuit Breaker."""
    
    def __init__(self, llm: Runnable, breaker: ProviderCircuitBreaker):
        self.llm = llm
        self.breaker = breaker
        
    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> Any:
        if not self.breaker.allow_request():
            logger.error("[CircuitBreaker] Fast-failing request to %s (Circuit is OPEN)", self.breaker.provider_name)
            return None
            
        try:
            result = self.llm.invoke(input, config=config, **kwargs)
            self.breaker.record_success()
            return result
        except Exception as e:
            self.breaker.record_failure()
            # Still raise the exception so the caller (or retry wrapper) can handle it
            raise e

# Global registry for per-provider circuit breakers.
_BREAKER_REGISTRY: dict[str, ProviderCircuitBreaker] = {}

def get_circuit_breaker(provider_name: str) -> ProviderCircuitBreaker:
    """Retrieve or create a circuit breaker for the given provider."""
    if provider_name not in _BREAKER_REGISTRY:
        _BREAKER_REGISTRY[provider_name] = ProviderCircuitBreaker(provider_name)
    return _BREAKER_REGISTRY[provider_name]
