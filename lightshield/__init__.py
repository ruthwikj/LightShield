"""Public LightShield middleware API."""

from __future__ import annotations

from typing import Any

from .adapters.anthropic_adapter import AnthropicAdapter
from .adapters.openai_adapter import LightShieldAdapter
from .adapters.langchain_adapter import LangChainAdapter


class LightShield:
    """Entry point for integrating the LightShield firewall.

    Example
    -------
    >>> from lightshield import LightShield
    >>> shield = LightShield()
    >>> response = shield.wrap(openai_client).chat.completions.create(
    ...     model="gpt-4",
    ...     messages=[{"role": "system", "content": "You are a helpful assistant"}],
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

    def wrap(self, client: Any) -> Any:
        """Wrap a provider client or LangChain chain with LightShield.

        The wrapper is detected based on simple duck-typing of commonly used
        client attributes (e.g. ``chat``, ``messages``).
        """

        # OpenAI-style client: has a ``chat`` namespace.
        if hasattr(client, "chat"):
            return LightShieldAdapter(client, raise_on_violation=self.raise_on_violation)

        # Anthropic-style client: has a ``messages`` namespace.
        if hasattr(client, "messages"):
            return AnthropicAdapter(client, raise_on_violation=self.raise_on_violation)

        # Fallback: assume this is a LangChain chain.
        return LangChainAdapter(client)


__all__ = ["LightShield"]
