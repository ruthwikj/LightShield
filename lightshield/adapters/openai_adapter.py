"""OpenAI SDK adapter for LightShield."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..core.encapsulator import Message, prepare_encapsulated_messages
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


class _ChatCompletionsProxy:
    """Proxy for ``client.chat.completions`` that injects LightShield logic."""

    def __init__(self, client: Any, adapter: "LightShieldAdapter") -> None:
        self._client = client
        self._adapter = adapter

    def create(self, *args: Any, **kwargs: Any) -> LightShieldResponse:
        """Intercept ``chat.completions.create`` calls."""

        messages: List[Message] = kwargs.get("messages") or []
        encapsulated, engine = prepare_encapsulated_messages(messages)
        kwargs = dict(kwargs)
        kwargs["messages"] = encapsulated

        raw_response = self._client.create(*args, **kwargs)
        validation = self._adapter._validate(raw_response, messages, encapsulated, engine)
        if self._adapter.raise_on_violation and not validation.is_safe:
            raise LightShieldViolationException(
                message="LightShield detected an unsafe response.",
                violations=validation.violations,
                confidence=validation.confidence,
            )
        return LightShieldResponse(raw=raw_response, lightshield=validation)


class _ChatProxy:
    """Proxy for ``client.chat`` that exposes a ``completions`` namespace."""

    def __init__(self, client: Any, adapter: "LightShieldAdapter") -> None:
        self.completions = _ChatCompletionsProxy(client.chat.completions, adapter)


class LightShieldAdapter:
    """Adapter that wraps an OpenAI client instance.

    This class is not used directly by developers; it is constructed by
    :class:`~lightshield.LightShield` via :meth:`LightShield.wrap`.
    """

    def __init__(self, client: Any, *, raise_on_violation: bool = False) -> None:
        self._client = client
        self.raise_on_violation = raise_on_violation
        self.chat = _ChatProxy(client, self)

    # Public API ---------------------------------------------------------

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


__all__ = ["LightShieldAdapter", "LightShieldResponse"]
