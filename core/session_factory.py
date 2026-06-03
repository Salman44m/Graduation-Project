import os
import logging
from typing import Any, Tuple

from config import settings, _circuit_breaker, _resolve_api_key
from core.llm_factory import create_chat_model, create_target_adapter, LLMFactoryError, MissingAPIKeyError

logger = logging.getLogger(__name__)

class SessionLLMFactory:
    """
    Unified LLM Factory. 
    Accepts explicit overrides (e.g. from API request kwargs), falls back to 
    environment variables in config.settings, and uses core.llm_factory to instantiate.
    Returns (attacker_llm, judge_llm, summariser_llm, target_adapter).
    """
    
    def __init__(self,
                 dry_run: bool = False,
                 attacker_provider: str | None = None,
                 attacker_model: str | None = None,
                 target_provider: str | None = None,
                 target_model: str | None = None):
                 
        self.dry_run = dry_run or settings.dry_run
        
        # Attacker config
        self.attacker_provider = attacker_provider or settings.attacker_provider
        self.attacker_model = attacker_model or settings.attacker_model
        
        # Target config
        self.target_provider = target_provider or settings.target_provider
        self.target_model = target_model or settings.target_model
        
        
    def _build_model_safe(self, provider: str, model: str, temperature: float, role: str) -> Any:
        key = _resolve_api_key(provider)
        if not key:
            return None
        try:
            return create_chat_model(provider=provider, model_name=model, temperature=temperature, api_key=key)
        except LLMFactoryError as exc:
            logger.warning("[SessionFactory] Failed to build %s model %s: %s", provider, model, exc)
            return None

    def _auto_detect_provider_and_build(self, provider_hint: str, model_hint: str, temperature: float, role: str) -> Any:
        # 1. Try requested provider
        if provider_hint:
            prov = provider_hint.lower()
            if not _resolve_api_key(prov):
                key_map = {"openai": "OPENAI_API_KEY", "deepseek": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "groq": "GROQ_API_KEY"}
                key_name = key_map.get(prov, f"{prov.upper()}_API_KEY")
                raise MissingAPIKeyError(f"No API key found for {prov}. Please add {key_name} to your .env file and restart.")
                
            if _circuit_breaker.is_open(prov):
                logger.warning("[CircuitBreaker] Skipping %s — circuit is OPEN.", provider_hint)
            else:
                llm = self._build_model_safe(prov, model_hint, temperature, role)
                if llm:
                    _circuit_breaker.record_success(prov)
                    return llm
                _circuit_breaker.record_failure(prov)

        # 2. Fallback chain
        from config import ATTACKER_MODEL, DEFAULT_MODEL, JUDGE_MODEL
        for prov, mdl in [("deepseek", ATTACKER_MODEL), ("openai", DEFAULT_MODEL), ("anthropic", JUDGE_MODEL), ("groq", "llama-3.3-70b-versatile")]:
            if prov == (provider_hint or "").lower() or _circuit_breaker.is_open(prov): continue
            if not _resolve_api_key(prov): continue
            m = model_hint or mdl
            
            llm = self._build_model_safe(prov, m, temperature, role)
            if llm:
                _circuit_breaker.record_success(prov)
                logger.info("[SessionFactory] %s LLM failover: %s/%s", role, prov, m)
                return llm
            _circuit_breaker.record_failure(prov)

        logger.warning("[SessionFactory] All providers exhausted or circuit-open for %s LLM.", role)
        return None

    def build(self) -> Tuple[Any, Any, Any, Any]:
        """Returns (attacker, judge, summariser, target)"""
        
        # ── DRY RUN ───────────────────────────────────────────────────────────
        if self.dry_run:
            class MockChatModel:
                def __init__(self, content): self.content = content
                def invoke(self, *args, **kwargs):
                    from langchain_core.messages import AIMessage
                    return AIMessage(content=self.content)
                def with_structured_output(self, *args, **kwargs):
                    return self
            
            from adapters.base_adapter import MockTargetAdapter
            
            return (
                MockChatModel("[DRY RUN] attacker payload"),
                MockChatModel("1.0\n[DRY RUN] judge rationale"),
                MockChatModel("[DRY RUN] compressed context"),
                MockTargetAdapter(
                    responses=[
                        "I understand your request. Let me think through this carefully and "
                        "provide you with a thorough academic analysis of the topic.",
                        "I'm sorry, I'm unable to assist with that specific request as it "
                        "conflicts with my operational guidelines.",
                        "That is an interesting technical scenario. From an educational "
                        "standpoint, here is what I can share about the general topic.",
                        "I appreciate the detailed context. Unfortunately, I cannot provide "
                        "the specific information you are requesting in this format.",
                    ],
                    model_id=self.target_model or "mock-target",
                )
            )

        # ── ATTACKER ──────────────────────────────────────────────────────────
        attacker = self._auto_detect_provider_and_build(
            self.attacker_provider,
            self.attacker_model,
            settings.attacker_temperature,
            "Attacker"
        )
        
        # ── JUDGE ─────────────────────────────────────────────────────────────
        judge = None
        if settings.judge_provider:
            judge = self._auto_detect_provider_and_build(
                settings.judge_provider,
                settings.judge_model,
                0.1,
                "Judge"
            )
        if judge is None:
            logger.debug("[SessionFactory] No dedicated judge LLM — sharing attacker LLM.")
            judge = attacker
            
        # ── SUMMARISER ────────────────────────────────────────────────────────
        summariser = None
        if settings.summariser_provider:
            summariser = self._auto_detect_provider_and_build(
                settings.summariser_provider,
                settings.summariser_model,
                0.3,
                "Summariser"
            )
        if summariser is None:
            summariser = attacker
            
        # ── TARGET ADAPTER ────────────────────────────────────────────────────
        target = None
        # In main.py, _TARGET_ADAPTER is passed explicitly, but we handle it via config now
        if self.target_provider:
            t_prov = self.target_provider.lower()
            key = None
            if t_prov == "deepseek": key = settings.target_openai_key or settings.openai_api_key
            elif t_prov == "openai": key = settings.target_openai_key or settings.openai_api_key
            elif t_prov == "anthropic": key = settings.target_anthropic_key or settings.anthropic_api_key
            elif t_prov == "groq": key = settings.target_groq_key or settings.groq_api_key
            
            # ollama handles its own base_url, doesn't need a key
            if t_prov != "ollama" and not key:
                key_map = {"openai": "TARGET_OPENAI_API_KEY", "deepseek": "TARGET_OPENAI_API_KEY", "anthropic": "TARGET_ANTHROPIC_API_KEY", "groq": "TARGET_GROQ_API_KEY"}
                key_name = key_map.get(t_prov)
                raise MissingAPIKeyError(f"No API key found for {t_prov}. Please add {key_name} to your .env file and restart.")
                
            try:
                target = create_target_adapter(
                    provider=t_prov,
                    model_name=self.target_model or "mock-target",
                    api_key=key,
                    base_url=settings.ollama_base_url if t_prov == "ollama" else None
                )
            except LLMFactoryError as exc:
                logger.error("[SessionFactory] Target adapter init failed: %s", exc)

        return (attacker, judge, summariser, target)
