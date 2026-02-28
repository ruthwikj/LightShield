"""Anthropic SDK adapter for LightShield."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..core.encapsulator import Message, prepare_encapsulated_messages
from ..core.hierarchy import TrustLevel
from ..core.uuid_engine import SessionUUIDEngine
from ..exceptions import LightShieldViolationException
from ..validators.output_validator import ValidationResult, validate_response


@dataclass
class LightShieldResponse:
    """Wrapper around an LLM SDK response that carries validation metadata."""

    raw: Any
    lightshield: ValidationResult

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self.raw, item)

    def __getitem__(self, item: Any) -> Any:  # pragma: no cover - passthrough
        return self.raw[item]


class _MessagesProxy:
    """Proxy for ``client.messages`` that injects LightShield logic."""

    def __init__(self, client: Any, adapter: "AnthropicAdapter") -> None:
        self._client = client
        self._adapter = adapter

    def create(self, *args: Any, **kwargs: Any) -> LightShieldResponse:
        """Intercept ``messages.create`` calls."""

        messages: List[Message] = kwargs.get("messages") or []
        encapsulated, engine = prepare_encapsulated_messages(messages)
        kwargs = dict(kwargs)
        kwargs["messages"] = encapsulated

        raw_response = self._client.create(*args, **kwargs)
        validation = self._adapter._validate(raw_response, messages, encapsulated, engine)
        if self._adapter.raise_on_violation and not validation.is_safe:
            content = _extract_anthropic_content(raw_response)
            raise LightShieldViolationException(
                message="LightShield detected an unsafe response.",
                violations=validation.violations,
                confidence=validation.confidence,
                violated_uuid_level=TrustLevel.USER,
                triggering_pattern=validation.violations[0] if validation.violations else None,
                raw_content=content[:500] if content else None,
                recommended_action="Review response content and input messages for injection attempts.",
            )
        return LightShieldResponse(raw=raw_response, lightshield=validation)


def _extract_anthropic_content(response: Any) -> str:
    """Extract text from Anthropic API response."""
    try:
        if hasattr(response, "content"):
            parts = []
            for b in (response.content or []):
                t = getattr(b, "text", None) if hasattr(b, "text") else (b.get("text") if isinstance(b, dict) else None)
                if isinstance(t, str):
                    parts.append(t)
            return "\n".join(parts)
    except Exception:
        pass
    return str(response)


class AnthropicAdapter:
    """Adapter that wraps an Anthropic client instance."""

    def __init__(self, client: Any, *, raise_on_violation: bool = False) -> None:
        self._client = client
        self.raise_on_violation = raise_on_violation
        # Anthropic's Python SDK exposes a ``messages`` namespace.
        self.messages = _MessagesProxy(client.messages, self)

    def _validate(
        self,
        response: Any,
        original_messages: List[Message],
        encapsulated_messages: List[Message],
        engine: SessionUUIDEngine,
    ) -> ValidationResult:
        """Run the core validation pipeline for a single response."""

        return validate_response(
            response,
            engine=engine,
            original_messages=original_messages,
            encapsulated_messages=encapsulated_messages,
        )


__all__ = ["AnthropicAdapter", "LightShieldResponse"]
