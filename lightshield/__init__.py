"""Public LightShield middleware API."""

from __future__ import annotations

from typing import Any, Optional

from .adapters.anthropic_adapter import AnthropicAdapter
from .adapters.langchain_adapter import LangChainAdapter
from .adapters.openai_adapter import LightShieldAdapter
from .adapters.pinecone_adapter import LightShieldRAGResult, sanitize_pinecone
from .core.uuid_engine import SessionUUIDEngine


class LightShield:
    """Entry point for integrating the LightShield firewall.

    Example
    -------
    >>> from lightshield import LightShield
    >>> shield = LightShield()
    >>> safe_docs = shield.sanitize_pinecone(pinecone_results)
    >>> response = shield.wrap(openai_client).chat.completions.create(
    ...     model="gpt-4",
    ...     messages=[{"role": "retrieved", "content": c["content"]} for c in safe_docs.safe_chunks],
    ... )
    >>> response.lightshield.is_safe
    True
    """

    def __init__(self, *, raise_on_violation: bool = False) -> None:
        """Create a new LightShield instance.

        Parameters
        ----------
        raise_on_violation:
            If True, adapters will raise :class:`LightShieldViolationException`
            when a high-confidence violation is detected instead of returning
            a potentially unsafe response.
        """

        self.raise_on_violation = raise_on_violation

    def sanitize_pinecone(
        self,
        raw_results: Any,
        *,
        engine: Optional[SessionUUIDEngine] = None,
        raise_on_high_risk: bool = False,
        risk_threshold: float = 0.8,
    ) -> LightShieldRAGResult:
        """Sanitize Pinecone query results at retrieval time.

        Scans chunks and metadata for instruction-like patterns, quarantines
        suspicious chunks, and UUID-tags safe chunks. Use before context assembly.

        Parameters
        ----------
        raw_results:
            Raw Pinecone query response.
        engine:
            Optional SessionUUIDEngine to reuse across sanitize + wrap.
        raise_on_high_risk:
            If True and risk_score >= risk_threshold, raises.
        risk_threshold:
            Threshold for raise_on_high_risk.

        Returns
        -------
        LightShieldRAGResult
            safe_chunks, flagged_chunks, risk_score, engine.
        """

        return sanitize_pinecone(
            raw_results,
            engine=engine,
            raise_on_high_risk=raise_on_high_risk,
            risk_threshold=risk_threshold,
        )

    def wrap(self, client: Any) -> Any:
        """Wrap a provider client or LangChain chain with LightShield.

        The wrapper is detected based on simple duck-typing of commonly used
        client attributes (e.g. ``chat``, ``messages``).
        """

        if hasattr(client, "chat"):
            return LightShieldAdapter(client, raise_on_violation=self.raise_on_violation)
        if hasattr(client, "messages"):
            return AnthropicAdapter(client, raise_on_violation=self.raise_on_violation)
        return LangChainAdapter(client)


__all__ = ["LightShield", "LightShieldRAGResult", "sanitize_pinecone"]
