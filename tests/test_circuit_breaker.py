import pytest
import time
from core.circuit_breaker import ProviderCircuitBreaker, CircuitState, CircuitBreakerRunnable
from langchain_core.runnables import RunnableLambda

def test_circuit_breaker_initial_state():
    cb = ProviderCircuitBreaker("test_provider", failure_threshold=3, recovery_timeout=0.1)
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request() is True

def test_circuit_breaker_opens_on_threshold():
    cb = ProviderCircuitBreaker("test_provider", failure_threshold=3, recovery_timeout=0.1)
    
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request() is True
    
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.allow_request() is False

def test_circuit_breaker_half_open_transition():
    cb = ProviderCircuitBreaker("test_provider", failure_threshold=1, recovery_timeout=0.1)
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    
    # Still open immediately
    assert cb.allow_request() is False
    
    # Wait for timeout
    time.sleep(0.15)
    
    # Should transition to HALF_OPEN
    assert cb.allow_request() is True
    assert cb.state == CircuitState.HALF_OPEN

def test_circuit_breaker_half_open_to_closed_on_success():
    cb = ProviderCircuitBreaker("test_provider", failure_threshold=1, recovery_timeout=0.1)
    cb.record_failure()
    time.sleep(0.15)
    cb.allow_request() # Transitions to HALF_OPEN
    
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_request() is True

def test_circuit_breaker_half_open_to_open_on_failure():
    cb = ProviderCircuitBreaker("test_provider", failure_threshold=1, recovery_timeout=0.1)
    cb.record_failure()
    time.sleep(0.15)
    cb.allow_request() # Transitions to HALF_OPEN
    
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.allow_request() is False

def test_circuit_breaker_runnable_success():
    cb = ProviderCircuitBreaker("test_provider")
    
    def fake_llm(x):
        return x + 1
        
    runnable = CircuitBreakerRunnable(RunnableLambda(fake_llm), cb)
    result = runnable.invoke(5)
    
    assert result == 6
    assert cb.state == CircuitState.CLOSED

def test_circuit_breaker_runnable_failure():
    cb = ProviderCircuitBreaker("test_provider", failure_threshold=1)
    
    def fake_llm(x):
        raise ValueError("Simulated outage")
        
    runnable = CircuitBreakerRunnable(RunnableLambda(fake_llm), cb)
    
    with pytest.raises(ValueError):
        runnable.invoke(5)
        
    assert cb.state == CircuitState.OPEN
    
    # Next call should fast-fail and return None without calling fake_llm
    result = runnable.invoke(5)
    assert result is None
